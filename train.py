# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from utils.attack_utils import targeted_attack_on_reflectance
from PIL import Image
from torchvision.utils import save_image
from utils.augmentations import preprocess
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=4, type=int, # server 上为3
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='/home/share/lowdetect/model/myWORK/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank',
                    type=int,
                    help='local rank for dist')

# 单独训练检测攻击
parser.add_argument('--TOG_all', action='store_true', help='train disturbance on night detection only')
parser.add_argument('--Save_perturb_dir', type=str, default='/home/share/lowdetect/dataset/myWORK/', help='Directory to save 昼夜扰动 images')

# 单独训练暗亮图检测
parser.add_argument('--DAY_Detection', action='store_true', help='train day detection only')
parser.add_argument('--NIGHT_Detection', action='store_true', help='train night detection only')

# 原始Domain adaptation
parser.add_argument('--DA', action='store_true', help='原始的Domain Adaptation')

args = parser.parse_args()
global local_rank
local_rank = args.local_rank


# 判断是否进行完整训练
DAY_Detection = args.DAY_Detection
NIGHT_Detection = args.NIGHT_Detection
TOG_all = args.TOG_all
Domain_adaptation = args.DA

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if torch.cuda.is_available():
    if args.cuda:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        import torch.distributed as dist

        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % gpu_num)
        dist.init_process_group('nccl')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True)
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=0,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True)

min_loss = np.inf

def train():
    per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    start_epoch = 0
    iteration = 0
    step_index = 0

    # 配置检测网络dsfd net
    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))

    if TOG_all and (not Domain_adaptation):
        weight_name = 'dsfd_day_best.pth'
        weight_path = os.path.join(args.save_folder, 'dark', weight_name)
        net_day = build_net('train', num_classes=cfg.NUM_CLASSES, model='dark')
        net_day.eval()
        net_day.load_state_dict(torch.load(weight_path))
        if local_rank == 0:
            print(f'Training disturbance only, loading {weight_path}...')

        weight_name = 'dsfd_night_best.pth'
        weight_path = os.path.join(args.save_folder, 'dark', weight_name)
        net_night = build_net('train', num_classes=cfg.NUM_CLASSES, model='dark')
        net_night.eval()
        net_night.load_state_dict(torch.load(weight_path))
        if local_rank == 0:
            print(f'Training disturbance only, loading {weight_path}...')
    
    # 中断恢复
    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    elif TOG_all and (not Domain_adaptation):
        start_epoch = cfg.EPOCHES - 1
    else:
        base_weights = torch.load(args.save_folder + basenet)
        if local_rank == 0:
            print('Load base network {}'.format(args.save_folder + basenet))
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)
            
        if local_rank == 0:
            print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the lr
    # 设置了根据批次大小和gpu数量调整学习率的机制
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.cuda:
        if args.multigpu:
            # 采用数据并行模型，多gpu
            net = torch.nn.parallel.DistributedDataParallel(net.cuda(), find_unused_parameters=True)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
            if TOG_all and (not Domain_adaptation):
                net_night = torch.nn.parallel.DistributedDataParallel(net_night.cuda(), find_unused_parameters=True)
                net_day = torch.nn.parallel.DistributedDataParallel(net_day.cuda(), find_unused_parameters=True)
        # net = net.cuda()
        cudnn.benchmark = True

    criterion = MultiBoxLoss(cfg, args.cuda)
    criterion_enhance = EnhanceLoss()
    if local_rank == 0:
        print('Loading wider dataset...')
        print('Using the specified args:')
        print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
    net_enh.eval()
    net.train()
    corr_mat = None

    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        loss_l1 = 0
        loss_c1 = 0
        loss_l2 = 0
        loss_c2 = 0
        loss_mu = 0
        loss_en = 0

        # 原始代码
        # for batch_idx, (images, targets, img_paths) in enumerate(train_loader):
        # 针对TOG all & Domain adaptation
        for batch_idx, (images, I_light, R_light, I_dark, R_dark, targets, img_paths) in enumerate(train_loader):
        
            # print(f"this info is: {img_paths}")
            # print( len( train_loader ) )
            # exit()
            # print( f"#batch {batch_idx} is working" )
            # print( "Before net:" )
            # print( f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB" )
            # print( f"Cached:    {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB" )

            with torch.no_grad():
                images = images.cuda() / 255.
                targetss = [ann.cuda() for ann in targets]
                # import sys; print(sys._getframe().f_lineno)
                # print(f"TOG is {TOG_all}")
                # exit()
            if Domain_adaptation or TOG_all:
                img_dark = torch.empty(size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3])).cuda()
                # Generation of degraded data and AET groundtruth
                for i in range(images.shape[0]):
                    with torch.no_grad():
                        img_dark[i], _ = Low_Illumination_Degrading(images[i])#ISP方法生成低照度图像
                        # import sys; print(sys._getframe().f_lineno)
                        
            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # 前向传播两个分支
            t0 = time.time()
            if TOG_all or Domain_adaptation:
                R_dark_gt, I_dark = net_enh(img_dark)
                R_light_gt, I_light = net_enh(images)
                # import sys; print(sys._getframe().f_lineno)
                
            if TOG_all and (not Domain_adaptation):
                net_day.eval()  
                net_night.eval()  # 在生成扰动时，冻结检测网络参数
                # ===== 添加扰动生成伪真值 =====
                # 可选条件：仅当 epoch 大于某个值，或每隔几轮，或损失较低时触发
                # if (epoch % 2 == 0) and local_rank == 0:  # 示例：仅 rank0 执行，避免多卡冲突
                if True:
                    adv_light_list = []
                    adv_dark_list = []
                    for i in range(R_light_gt.shape[0]):   # batch_size 次
                        # import sys; print(sys._getframe().f_lineno)
                        # 对亮图扰动
                        if local_rank == 0:
                            print(f"[Batch {batch_idx}] Processing light image {i+1}/{R_light_gt.shape[0]}...")
                        # 1. 对亮图反射图添加扰动（完全独立的另一组）
                        adv_l = targeted_attack_on_reflectance(
                            net=net_day,
                            criterion=criterion,
                            ref_img=R_light_gt[i:i+1],
                            targets=[targetss[i]],  # 标注共用
                            eps=4/255.,
                            eps_iter=0.5/255.,
                            n_iter=10
                        )
                        adv_light_list.append(adv_l)
                        # 对暗图扰动
                        if local_rank == 0:
                            print(f"[Batch {batch_idx}] Processing dark image {i+1}/{R_dark_gt.shape[0]}...")
                        # 1. 对亮图反射图添加扰动（完全独立的另一组）
                        adv_d = targeted_attack_on_reflectance(
                            net=net_night,
                            criterion=criterion,
                            ref_img=R_dark_gt[i:i+1],
                            targets=[targetss[i]],  # 标注共用
                            eps=4/255.,
                            eps_iter=0.5/255.,
                            n_iter=10
                        )
                        adv_dark_list.append(adv_d)

                    R_light_gt = torch.cat(adv_light_list, dim=0)
                    R_dark_gt = torch.cat(adv_dark_list, dim=0)
                    net_day.train()
                    net_night.train()

                    # 保存扰动后的图像
                    save_illum = os.path.join(args.Save_perturb_dir, 'illuminDay')
                    save_pertur = os.path.join(args.Save_perturb_dir, 'perturbDay')
                    os.makedirs(save_illum, exist_ok=True)
                    os.makedirs(save_pertur, exist_ok=True)
                    for i in range(R_light_gt.shape[0]):
                        # 获取原始文件名（不含目录）
                        orig_path = img_paths[i]
                        base_name = os.path.basename(orig_path)   # 例如 "000001.jpg"
                        # 保留原始扩展名
                        # 保存光照亮图
                        save_path = os.path.join(save_illum, base_name)
                        img = I_light[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)
                        # 保存扰动亮图
                        save_path = os.path.join(save_pertur, base_name)
                        img = R_light_gt[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)

                    save_illum = os.path.join(args.Save_perturb_dir, 'illuminNight')
                    save_pertur = os.path.join(args.Save_perturb_dir, 'perturbNight')
                    os.makedirs(save_illum, exist_ok=True)
                    os.makedirs(save_pertur, exist_ok=True)
                    for i in range(R_dark_gt.shape[0]):
                        # 获取原始文件名（不含目录）
                        orig_path = img_paths[i]
                        base_name = os.path.basename(orig_path)   # 例如 "000001.jpg"
                        # 保留原始扩展名
                        # 保存光照亮图
                        save_path = os.path.join(save_illum, base_name)
                        img = I_dark[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)
                        # 保存扰动亮图
                        save_path = os.path.join(save_pertur, base_name)
                        img = R_dark_gt[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)

                # 若使用多卡，需广播到其他 rank
                # if args.multigpu:
                #     dist.broadcast(R_light_gt, src=0)

                continue
        if TOG_all and (not Domain_adaptation):
            return 0
                # ===== 新增结束 =====
        if True:
            if DAY_Detection:
                inputs = R_light_gt.detach()
            elif NIGHT_Detection:
                inputs = R_dark_gt.detach()

            if Domain_adaptation:
                out, out2, loss_mutual = net(img_dark, images, I_dark.detach(), I_light.detach())
                R_dark, R_light, R_dark_2, R_light_2 = out2
            elif DAY_Detection or NIGHT_Detection:
                out= net.module.forward_detection(inputs)
                R_dark, R_light = out

                # print( "After net:" )
                # print( f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB" )
                # print( f"Cached:    {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB" )

            # backprop
            optimizer.zero_grad()
            # 损失函数整理
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            if Domain_adaptation:
                loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
                loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                            1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))

            if False:
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
                def GradientLoss(pred, target):
                    out = 0
                    for i in range(3):
                        pred_channel = pred[:, i:i+1, :, :]  # 获取第i个通道
                        target_channel = target[:, i:i+1, :, :]  # 获取第i个通道

                        pred_grad_x = F.conv2d(pred_channel, sobel_x.to(pred.device), padding=1)
                        pred_grad_y = F.conv2d(pred_channel, sobel_y.to(pred.device), padding=1)
                        target_grad_x = F.conv2d(target_channel, sobel_x.to(target.device), padding=1)
                        target_grad_y = F.conv2d(target_channel, sobel_y.to(target.device), padding=1)
                        
                        out += torch.mean(torch.abs(pred_grad_x - target_grad_x)) + torch.mean(torch.abs(pred_grad_y - target_grad_y))
                    return out
            
                loss_enhance = (1 * GradientLoss(R_dark, R_dark_gt.detach()) # 强调轮廓
                                # + self.gram_loss(R_dark, R_dark_2.detach()) # 强调纹理
                                + (1. - ssim(R_dark, R_dark_gt.detach()))  # 强调结构
                                ) 
                loss_enhance2 = (1 * GradientLoss(R_light, R_light_gt.detach())
                                # + self.gram_loss(R_light, R_light_2.detach())
                                + (1. - ssim(R_light, R_light_gt.detach()))
                                )
                
            loss += loss_enhance2 + loss_enhance + loss_mutual

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
            optimizer.step()
            t1 = time.time()
            losses += loss.item()
            loss_l1 += loss_l_pa1l.item()
            loss_c1 += loss_c_pal1.item()
            loss_l2 += loss_l_pa12.item()
            loss_c2 += loss_c_pal2.item()
            if Domain_adaptation:
                loss_mu += loss_mutual.item()
                loss_en += loss_enhance.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                tloss_l1 = loss_l1 / (batch_idx + 1)
                tloss_c1 = loss_c1 / (batch_idx + 1)
                tloss_l2 = loss_l2 / (batch_idx + 1)
                tloss_c2 = loss_c2 / (batch_idx + 1)
                if Domain_adaptation:
                    tloss_mu = loss_mu / (batch_idx + 1)
                    tloss_en = loss_en / (batch_idx + 1)
                
                if local_rank == 0:
                    print( 'Timer: %.4f' % (t1 - t0) )
                    print( 'epoch:' + repr( epoch ) + ' || iter:' + repr( iteration ) + ' || Loss:%.4f' % (tloss) )
                    print( '->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format( tloss_c1 , tloss_l1 ) )
                    print( '->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format( tloss_c2 , tloss_l2 ) )
                    if Domain_adaptation:
                        print( '->> mutual loss:{:.4f} || enhanced loss:{:.4f}'.format( tloss_mu , tloss_en ) )
                    print( '->>lr:{}'.format( optimizer.param_groups[ 0 ][ 'lr' ] ) )
        
            if iteration != 0 and iteration % 5000 == 0:
                if local_rank == 0:
                    print('Saving state, iter:', iteration)
                    if DAY_Detection:
                        file = 'dsfd_day_' + repr(iteration) + '.pth'
                    elif NIGHT_Detection:
                        file = 'dsfd_night_' + repr(iteration) + '.pth'
                    elif Domain_adaptation:
                        file = 'dsfd_da_' + repr(iteration) + '.pth'
                    torch.save(dsfd_net.state_dict(),
                               os.path.join(save_folder, file))
            iteration += 1
        # if local_rank == 0:
        if (epoch + 1) >= 0:
            if not (TOG_Day or TOG_Night):
                val(epoch, net, dsfd_net, net_enh, criterion)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, net_enh, criterion):
    net.eval()
    net_enh.eval()
    step = 0
    losses = torch.tensor(0.).cuda()
    losses_enh = torch.tensor(0.).cuda()
    t1 = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
            with torch.no_grad():
                if args.cuda:
                    images = images.cuda() / 255.
                    targets = [ann.cuda() for ann in targets]
                else:
                    images = images / 255.
                    targets = [ann.cuda() for ann in targets]
                if Domain_adaptation:
                    images = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])],
                                    dim=0)
                if DAY_Detection or NIGHT_Detection:
                    images,_ = net_enh(images)
                out = net.module.test_forward(images)

            # loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            # import sys; print(sys._getframe().f_lineno)
            loss_l_pa12, loss_c_pal2 = criterion(out, targets)
            loss = loss_l_pa12 + loss_c_pal2

            losses += loss.item()
            step += 1
        dist.reduce(losses, 0, op=dist.ReduceOp.SUM)

    tloss = losses / step / torch.cuda.device_count()
    t2 = time.time()
    if local_rank == 0:
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        if local_rank == 0:
            print('Saving best state,epoch', epoch)
            if DAY_Detection:
                torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd_day_best.pth'))
            elif NIGHT_Detection:
                torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd_night_best.pth'))
            # elif TOG_all:
            #     torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd_tog_best.pth'))
            elif Domain_adaptation:
                torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd_da_best.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    if local_rank == 0:
        if DAY_Detection:
            torch.save(states, os.path.join(save_folder, 'dsfd_day_checkpoint.pth'))
        elif NIGHT_Detection:
            torch.save(states, os.path.join(save_folder, 'dsfd_night_checkpoint.pth'))
        # elif TOG_all and (not Domain_adaptation):
        #     torch.save(states, os.path.join(save_folder, 'dsfd_tog_checkpoint.pth'))
        elif Domain_adaptation:
            torch.save(states, os.path.join(save_folder, 'dsfd_da_checkpoint.pth'))

def generate_retinex_data(save_retinex_dir):
    """生成低光退化+Retinex分解的结果，保存为图像"""
    # 初始化模型
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))
    if args.cuda:
        net_enh = net_enh.cuda()
    net_enh.eval()

    # 数据集
    path_day = os.path.join(save_retinex_dir, 'decom_DAY')
    os.makedirs(path_day, exist_ok=True)
    path_night = os.path.join(save_retinex_dir, 'decom_NIGHT')
    os.makedirs(path_night, exist_ok=True)
    
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)

    print(f"Generating Retinex data from train set...")
    for idx, (images, targets, img_paths) in enumerate(dataloader):
        with torch.no_grad():
            images = images.cuda() / 255.0 if args.cuda else images / 255.0
            # 低光退化
            img_dark, _ = Low_Illumination_Degrading(images[0])  # 单张 [C,H,W]
            img_dark = img_dark.unsqueeze(0)  # [1,C,H,W]
            # Retinex分解
            R_day, I_day = net_enh(images)
            R_night, I_night = net_enh(img_dark)

            base = os.path.splitext(os.path.basename(img_paths[0]))[0]
            ext = os.path.splitext(os.path.basename(img_paths[0]))[1]
            # 保存 (转为0~255 uint8)
            for name, tensor in zip(['reflectance', 'illumination'], [R, I]):
                # tensor [1,3,H,W] 或 [1,1,H,W]（光照图可能是单通道）
                # 如果是单通道，复制为3通道以便保存
                if tensor.shape[1] == 1:
                    tensor = tensor.repeat(1, 3, 1, 1)
                # 保存
                save_path = os.path.join(path_train, name, f"{base}{ext}")
                save_image(tensor, save_path)  # 自动反归一化
        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1} training images")

    print("Done!")

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

def load_img_from_dir(subdir, img_path, base_dir=args.Save_perturb_dir):
    """从子目录加载图像，返回 [0,1] tensor"""
    fname = os.path.basename(img_path)
    full_path = os.path.join(base_dir, subdir, fname)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing {full_path}")
    img = Image.open(full_path).convert('RGB')
    return TF.to_tensor(img)

def apply_sync_transforms_and_norm(images, img_dark, I_light, R_light_gt, I_dark, R_dark_gt, targets, cfg):
    """
    在 GPU 上同步对配对图像进行数据增强和格式归一化。
    images 等输入此时应为 (B, C, H, W) 格式的 GPU Tensor，且数值范围应为 0~1 (因为之前除了 255.)。
    """
    B = images.shape[0]
    device = images.device
    
    # 假设 cfg.img_mean 的原始顺序是 [B, G, R] (常见的 VGG 均值，如 104, 117, 123)
    # 因为我们当前的数据是 RGB 顺序，我们需要把均值倒过来 [R, G, B] 用于减法
    rgb_mean = [cfg.img_mean[2], cfg.img_mean[1], cfg.img_mean[0]]
    mean_tensor = torch.tensor(rgb_mean, device=device).view(1, 3, 1, 1)

    new_targets = []
    
    for i in range(B):
        bbox = targets[i]
        
        # ====== 1. 同步随机水平翻转 (Mirror: 50% 概率) ======
        if random.random() > 0.5:
            # 翻转所有对应的图像 (dims=[2] 表示在 Width 维度翻转)
            images[i] = torch.flip(images[i], dims=[2])
            
            if img_dark is not None:
                img_dark[i] = torch.flip(img_dark[i], dims=[2])
            if I_light is not None:
                I_light[i] = torch.flip(I_light[i], dims=[2])
            if R_light_gt is not None:
                R_light_gt[i] = torch.flip(R_light_gt[i], dims=[2])
            if I_dark is not None:
                I_dark[i] = torch.flip(I_dark[i], dims=[2])
            if R_dark_gt is not None:
                R_dark_gt[i] = torch.flip(R_dark_gt[i], dims=[2])
            
            # 同步更新人脸框 (x_min 和 x_max 互换并用 1 减，假设 bbox 坐标已归一化)
            if len(bbox) > 0:
                tmp_xmin = bbox[:, 0].clone()
                bbox[:, 0] = 1.0 - bbox[:, 2]
                bbox[:, 2] = 1.0 - tmp_xmin
                
        new_targets.append(bbox)
        
    # ====== 2. 目标检测器的归一化 ======
    # 【注意】你的 net_enh (Retinex) 可能需要 0~1 的数据，所以需要保留 0~1 的版本
    # 但检测网络 (DSFD/VGG) 需要的是 (img * 255 - mean) 且为 RGB 格式 (根据原 preprocess 逻辑)
    
    # 创建给检测网络专用 (det_xxx) 的归一化图像
    det_images = images * 255.0 - mean_tensor
    det_img_dark = None
    if img_dark is not None:
        det_img_dark = img_dark * 255.0 - mean_tensor
        
    return images, img_dark, I_light, R_light_gt, I_dark, R_dark_gt, det_images, det_img_dark, new_targets

if __name__ == '__main__':
    train()
