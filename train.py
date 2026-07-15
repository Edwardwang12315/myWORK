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
parser.add_argument('--TOG_Day', action='store_true', help='train disturbance on day detection only')
parser.add_argument('--TOG_Night', action='store_true', help='train disturbance on night detection only')

# 单独训练暗亮图检测
parser.add_argument('--DAY_Detection', action='store_true', help='train day detection only')
parser.add_argument('--NIGHT_Detection', action='store_true', help='train night detection only')

args = parser.parse_args()
global local_rank
local_rank = args.local_rank


# 判断是否进行完整训练
TOG_Day = args.TOG_Day
TOG_Night = args.TOG_Night
DAY_Detection = args.DAY_Detection
NIGHT_Detection = args.NIGHT_Detection
if (TOG_Day or TOG_Night or DAY_Detection or NIGHT_Detection):
    Domain_adaptation = False
else:
    Domain_adaptation = True


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

    # 中断恢复
    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weights = torch.load(args.save_folder + basenet)
        if local_rank == 0:
            print('Load base network {}'.format(args.save_folder + basenet))
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)
    if not args.resume:
        if TOG_Day or TOG_Night:
            start_epoch = cfg.EPOCHES - 1
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

        for batch_idx, (images, targets, img_paths) in enumerate(train_loader):
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
            if TOG_Night or NIGHT_Detection or Domain_adaptation:
                img_dark = torch.empty(size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3])).cuda()
                # Generation of degraded data and AET groundtruth
                for i in range(images.shape[0]):
                    img_dark[i], _ = Low_Illumination_Degrading(images[i])#ISP方法生成低照度图像

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # 前向传播两个分支
            t0 = time.time()
            if TOG_Night or NIGHT_Detection or Domain_adaptation:
                R_dark_gt, I_dark = net_enh(img_dark)
            if TOG_Day or DAY_Detection or Domain_adaptation:
                R_light_gt, I_light = net_enh(images)

            if TOG_Night:
                net.eval()  # 在生成扰动时，冻结检测网络参数
                # ===== 添加扰动生成伪真值 =====
                # 可选条件：仅当 epoch 大于某个值，或每隔几轮，或损失较低时触发
                # if (epoch % 2 == 0) and local_rank == 0:  # 示例：仅 rank0 执行，避免多卡冲突
                if True:  
                    # 对 dark 反射图进行攻击（也可对 light 攻击，根据需要）
                    adv_dark_list = []
                    for i in range(R_dark_gt.shape[0]):   # batch_size 次
                        # 1. 对暗图反射图添加扰动（独立）
                        adv_d = targeted_attack_on_reflectance(
                            net=net,
                            criterion=criterion,
                            ref_img=R_dark_gt[i:i+1],
                            targets=[targetss[i]],
                            eps=4/255.,
                            eps_iter=0.5/255.,
                            n_iter=10
                        )
                        adv_dark_list.append(adv_d)
                    R_dark_gt = torch.cat(adv_dark_list, dim=0)
                    net.train()

                    # 保存扰动后的图像
                    save_dir = os.path.join(save_folder, 'tog_night_results')
                    os.makedirs(save_dir, exist_ok=True)
                    for i in range(R_dark_gt.shape[0]):
                        # 获取原始文件名（不含目录）
                        orig_path = img_paths[i]
                        base_name = os.path.basename(orig_path)   # 例如 "000001.jpg"
                        # 保留原始扩展名
                        save_path = os.path.join(save_dir, base_name)
                        img = R_dark_gt[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)
                
                # 若使用多卡，需广播到其他 rank
                # if args.multigpu:
                #     dist.broadcast(R_dark_gt, src=0)
                
                continue

            elif TOG_Day:
                net.eval()  # 在生成扰动时，冻结检测网络参数
                # ===== 添加扰动生成伪真值 =====
                # 可选条件：仅当 epoch 大于某个值，或每隔几轮，或损失较低时触发
                # if (epoch % 2 == 0) and local_rank == 0:  # 示例：仅 rank0 执行，避免多卡冲突
                if True:
                    # 对 dark 反射图进行攻击（也可对 light 攻击，根据需要）
                    adv_light_list = []
                    for i in range(R_dark_gt.shape[0]):   # batch_size 次
                        # 2. 对亮图反射图添加扰动（完全独立的另一组）
                        adv_l = targeted_attack_on_reflectance(
                            net=net,
                            criterion=criterion,
                            ref_img=R_light_gt[i:i+1],
                            targets=[targetss[i]],  # 标注共用
                            eps=4/255.,
                            eps_iter=0.5/255.,
                            n_iter=10
                        )
                        adv_light_list.append(adv_l)

                    R_light_gt = torch.cat(adv_light_list, dim=0)
                    net.train()

                    # 保存扰动后的图像
                    save_dir = os.path.join(save_folder, 'tog_day_results')
                    os.makedirs(save_dir, exist_ok=True)
                    for i in range(R_light_gt.shape[0]):
                        # 获取原始文件名（不含目录）
                        orig_path = img_paths[i]
                        base_name = os.path.basename(orig_path)   # 例如 "000001.jpg"
                        # 保留原始扩展名
                        save_path = os.path.join(save_dir, base_name)
                        img = R_light_gt[i].cpu().detach()  # [3, H, W]
                        # 使用 torchvision 保存
                        save_image(img, save_path)

                # 若使用多卡，需广播到其他 rank
                # if args.multigpu:
                #     dist.broadcast(R_light_gt, src=0)

                continue

                # ===== 新增结束 =====

            if DAY_Detection:
                inputs = R_light_gt.detach()
            elif NIGHT_Detection:
                inputs = R_dark_gt.detach()
            elif TOG_Day:
                inputs = R_light_gt
            elif TOG_Night:
                inputs = R_dark_gt

            if Domain_adaptation:
                out, out2, loss_mutual = net(img_dark, images, I_dark.detach(), I_light.detach())
            else:
                out= net.module.forward_detection(inputs)

            
            if Domain_adaptation:
                # R_dark, R_light, R_dark_2, R_light_2 = out2
                R_dark, R_light = out2
                # print( "After net:" )
                # print( f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB" )
                # print( f"Cached:    {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB" )

            # backprop
            optimizer.zero_grad()
            # 损失函数整理
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2

            # loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
            # loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
            #             1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))

            if Domain_adaptation:
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
    step = 0
    losses = torch.tensor(0.).cuda()
    losses_enh = torch.tensor(0.).cuda()
    t1 = time.time()

    for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                images = images.cuda() / 255.
                targets = [ann.cuda() for ann in targets]
            else:
                images = images / 255.
                targets = [ann.cuda() for ann in targets]
        if TOG_Night or NIGHT_Detection or Domain_adaptation:
            images = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])],
                               dim=0)
        if DAY_Detection or NIGHT_Detection:
            images,_ = net_enh(images)
        out = net.module.test_forward(images)

        # loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
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
        elif Domain_adaptation:
            torch.save(states, os.path.join(save_folder, 'dsfd_da_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()
