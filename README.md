# Learning Loss for Active Learning

An unofficial PyTorch implementation of the paper [Learning Loss for Active Learning](https://arxiv.org/pdf/1905.03677.pdf).

### Requirements

```shell
torch
torchvision
matplotlib
tqdm
cv2
imageio
imutils
```

### Usage

```shell
python main.py --task {clf OR detection OR hpe}
```

-  task
    - ```clf```: image classification
    - ```detection```: object detection
    - ```hpe```: human pose estimation
    

#### image classification
- dataset: CIFAR10 & CIFAR100
- model: ResNet
- metric: Accuracy

```shell
python main.py --task clf --dataset CIFAR10 --subset 10000 --num_epoch 200 --batch_size 128 --lr 0.1 --epoch_loss 120 --weights 1.0 --milestone 160
```

#### object detection
- dataset: PASCAL VOC2007 & 2012
- model: SSD (Single Shot Multibox Detector)
- metric: mAP

```shell
python main.py --task detection --dataset VOC0712 --num_epoch 300 --batch_size 32 --lr 0.001 --epoch_loss 240 --weights 1.0 --milestone 240 
```

#### human pose estimation
- dataset: MPII
- model: SHN (Stacked Hourglass Networks)
- metric: PCKh@0.5

```shell
python main.py --task hpe --dataset mpii --subset 5000 --num_epoch 125 --batch_size 6 --wdecay 0 --lr 0.00025 --epoch_loss 75 --weights 0.0001 --milestone 100 
```

### References

- https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
- https://github.com/amdegroot/ssd.pytorch
- https://github.com/bearpaw/pytorch-pose
- https://github.com/kuangliu/pytorch-cifar





