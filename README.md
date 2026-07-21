# 构建更好的辅助分支

## 实验 2026.7.13
1. 在DAINet原版内容基础上，添加了TOG扰动
2. batch=4，lr=5e-4，momentum=0.9，weight decay=5e-4，gamma=0.1
3. CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py

## 实验 2026.7.14
1. 发现存在问题：TOG强制要求每次batch循环前，都要先进行10-20次训练损失，因为每张图像的扰动添加都是不相同的，训练效率很低下
2. 设计实验：设置独立扰动图像训练，对数据集每张图像（无论亮暗）单独进行Retinex分解，并针对预训练dsfd检测网络攻击，保存反射图、光照图、扰动图于本地；而后再进行主体实验，不再对单张图像执行分解和检测攻击，而是直接从本地读取。
3. 潜在风险1：理想中的TOG是使用最优的检测网络进行攻击，根据mean teacher架构，也可以退而求其次使用不断更新的检测网络进行攻击，效果仍能保留一部分，但是本实验设计仅使用预训练的检测网络进行检测攻击，得到的扰动图像难以保证质量。
4. 潜在风险2：dsfd本身没有跨域检测能力，对亮图进行检测攻击尚且能有质量保证，但是对暗图来说，dsfd难以维系正常检测功能。
5. 实验修改：对昼夜图像先分别训练一套最优的dsfd，然后分别训练扰动攻击
6. CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py 
7. 待补充--保存DAY NIGHT检测权重，保存TOG检测攻击的扰动图像
8. 进行中：CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --DAY_Detection --batch_size 6
9. 一直显存爆OOM，CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --DAY_Detection --batch_size 1 --num_workers 4
10. 外部分解所有数据集，使用Generate_data参数控制
11. 新变动：外部分解数据集的工作暂停，调小batch似乎仍然可以工作
12. batch=4起始占用13500MB，batch=6起始占用21300MB
13. batch=4进入到val计算误差的时候，显存爆了
14. 根据ai的说法，retinex使用enh.eval()只是不再随机dropout不再随机使用batchnormal，本质上仍然保留了计算痕迹，只是不触发自动求导。只有使用torch.no_grad()才能保证过程中不记录梯度信息和计算图。
15. batch=4稳定在17900MB附近 因为GPU3莫名奇妙空转
16. GPU2 | batch=6起始占用21300MB 稳定在22100MB
17. CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --DAY_Detection --batch_size 6 --num_worker 6
18. epoch:69 || iter:150200 || Loss:5.2209 
    ->> pal1 conf loss:1.5438 || pal1 loc loss:1.2234
    ->> pal2 conf loss:1.3793 || pal2 loc loss:1.0744
    ->>lr:6.123500000000002e-07
    Timer: 604.6541
    test epoch:69 || Loss:2.594
19. 对夜间图像进行训练
    CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --NIGHT_Detection --batch_size 6 --num_worker 6
20. TOG_Day训练
    CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py --TOG_Day --batch_size 6 --num_worker 6
21. TOG训练不能用于保存扰动，因为算法会提前对图像裁剪
22. 实验方案调整：保证DAINet最小变动，只添加TOG，并联合训练不搞单独训练
23. 具体方案：基于DAINet网络最小改动的条件下，借助昼夜反射图分别训练过的dsfd作为检测器，昼夜图像对输入到辅助分支后，经retinex分解得到反射图，通过TOG的检测扰动添加，找到检测最优扰动图。再以此扰动图为target，对齐主通路特征解码图。
24.  CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --TOG --batch_size 2 --num_worker 8
25.  太慢了，太慢了
26.  改，依旧是拆分开进行。先TOG day night，保存扰动图
     1.   取出原图，日间图像直接retinex分解，保存光照图在本地，反射图传入TOG扰动添加训练
     2.   夜间图先对原图退化，再retinex分解，保存光照图在本地，反射图传入TOG扰动添加训练
     3.   TOG使用对应领域的best权重，训练完成保存本地
     4.   CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --TOG_all --batch_size 8 --num_worker 0
27.  再取图（widerface&扰动图day&扰动图night），退化ISP，一起增强数据
     1.   取出原图,日间图像直接退化--img,img_dark
     2.   取出光照图和扰动反射图--illmin,illmin_dark,perturDay,perturNight
     3.   数据增强--img,img_dark,illmin,illmin_dark,perturDay,perturNight
     4.   img,img_dark输入dainet中
28.  img、imgdark、perturday、perturnight
29.  2026.7.19
30.  阅读Yildirim - 2026有想
31.  相位、幅值
32.  





## 


## 🔨 To-Do List

## :rocket: Installation

Begin by cloning the repository and setting up the environment:

```
git clone https://github.com/ZPDu/DAI-Net.git
cd DAI-Net

conda create -y -n dainet python=3.7
conda activate dainet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## :notebook_with_decorative_cover: Training


#### Data and Weight Preparation

- Download the WIDER Face Training & Validation images at [WIDER FACE](http://shuoyang1213.me/WIDERFACE/).
- Obtain the annotations of [training set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_train.txt) and [validation set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_val.txt).
- Download the [pretrained weight](https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view?usp=drive_link) of Retinex Decomposition Net.
- Prepare the [pretrained weight](https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view?usp=drive_link) of the base network.

Organize the folders as:

```
.
├── program
│   ├── myWORK
│   │   ├── ......
├── model
│   ├── myWORK
│   │   ├── decomp.pth
│   │   ├── vgg16_reducedfc.pth
├── dataset
│   ├── myWORK
│   │   ├── wider_face_train.txt
│   │   ├── wider_face_val.txt
│   │   ├── WiderFace
│   │   │   ├── WIDER_train
│   │   │   └── WIDER_val
```

#### Model Training

To train the model, run

```
CUDA_VISIBLE_DEVICES=x MASTER_PORT=xxx python -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPUS$ train.py
```

## :notebook: Evaluation​

On Dark Face:

- Download the testing samples from [UG2+ Challenge](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c).
- Download the checkpoints: [DarkFaceZSDA](https://drive.google.com/file/d/1BdkYLGo7PExJEMFEjh28OeLP4U1Zyx30/view?usp=drive_link) (28.0) or [DarkFaceFS](https://drive.google.com/file/d/1ykiyAaZPl-mQDg_lAclDktAJVi-WqQaC/view?usp=drive_link) (52.9, finetuned with full supervision).
- Set (1) the paths of testing samples & checkpoint, (2) whether to use a multi-scale strategy, and run test.py.
- Submit the results for benchmarking. ([Detailed instructions](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c)).

On ExDark:

- Our experiments are based on the codebase of [MAET](https://github.com/cuiziteng/ICCV_MAET). You only need to replace the checkpoint with [ours](https://drive.google.com/file/d/1g74-aRdQP0kkUe4OXnRZCHKqNgQILA6r/view?usp=drive_link) for evaluation.

# 调试记录
## 2025.1.22

## 2025.4.10


## 📑 Citation
