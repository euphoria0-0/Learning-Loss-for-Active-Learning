# Learning Loss for Active Learning

An unofficial PyTorch implementation of the paper [Learning Loss for Active Learning](https://arxiv.org/pdf/1905.03677.pdf).

### Requirements

```shell
torch
torchvision
tensorboardX
```

### Usage

```shell
python main.py --task {clf OR detection}
```

-  task
    - ```clf```: image classification
    - ```detection```: object detection
    

#### image classification
- dataset: CIFAR10 & CIFAR100
- model: ResNet
- metric: Accuracy

#### object detection
- dataset: PASCAL VOC2007 & 2012
- model: SSD (Single Shot Multibox Detector)
- metric: mAP




### References

- https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
- https://github.com/amdegroot/ssd.pytorch
- https://github.com/kuangliu/pytorch-cifar





