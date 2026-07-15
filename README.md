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
