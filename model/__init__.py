import os
import torch
from torch import nn

from .lossnet import LossNet
from .resnet import ResNet18
from .hourglass import hg
from .ssd_pytorch.ssd import build_ssd
from data.voc import voc_cfg
from utils.utils import Logger



def get_ssd_model(args, phase='train'):
    if phase == 'train':
        ssd_net = build_ssd('train', voc_cfg['min_dim'], args.nClass)

        if args.resume:
            ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load('./model/ssd_pytorch/vgg16_reducedfc.pth')
            ssd_net.vgg.load_state_dict(vgg_weights)
            # initialize
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
    else:
        ssd_net = build_ssd('test', voc_cfg['min_dim'], args.nClass)  # initialize SSD
        ssd_net.load_state_dict(torch.load('./model/ssd_pytorch/VOC.pth'))

    loss_net = LossNet(feature_sizes=[512, 1024, 512, 256, 256, 256],
                       num_channels=[512, 1024, 512, 256, 256, 256],
                       task='detection')

    ssd_net = ssd_net.to(args.device)
    loss_net = loss_net.to(args.device)

    return {'backbone': ssd_net, 'module': loss_net}


def get_resnet_model(args):
    resnet = ResNet18(args.nClass)
    loss_net = LossNet()

    resnet = resnet.to(args.device)
    loss_net = loss_net.to(args.device)

    return {'backbone': resnet, 'module': loss_net}


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def get_shn_model(args, model=None, optimizer=None):
    # optionally resume from a checkpoint
    title = 'mpii hg'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    shnet = hg(num_stacks=2, num_blocks=1, num_classes=args.nJoint, resnet_layers=50)
    loss_net = LossNet(feature_sizes=[64,64,256],
                       num_channels=[256,256],
                       task='hpe')

    shnet = shnet.to(args.device)
    loss_net = loss_net.to(args.device)

    return {'backbone': shnet, 'module': loss_net}

def model_parallel(model):
    model['backbone'] = torch.nn.DataParallel(model['backbone'])
    model['module'] = torch.nn.DataParallel(model['module'])
    torch.backends.cudnn.benchmark = True
    return model