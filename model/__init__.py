import torch
from torch import nn

from .lossnet import LossNet
from .ssd_pytorch.ssd import build_ssd
from .resnet import ResNet18
from data.voc import voc_cfg



def get_ssd_model(args, phase='train'):
    if phase == 'train':
        ssd_net = build_ssd('train', voc_cfg['min_dim'], args.nClass)

        if args.resume:
            ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load('./ssd_pytorch/weights/vgg16_reducedfc.pth')
            ssd_net.vgg.load_state_dict(vgg_weights)
            # initialize
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
    else:
        ssd_net = build_ssd('test', voc_cfg['min_dim'], args.nClass)  # initialize SSD
        ssd_net.load_state_dict(torch.load('./results/VOC.pth'))

    loss_net = LossNet(feature_sizes=[512, 1024, 512, 256, 256, 256],
                       num_channels=[512, 1024, 512, 256, 256, 256],
                       task='detection')

    ssd_net = ssd_net.to(args.device)
    loss_net = loss_net.to(args.device)

    if ',' in args.gpu_id:
        ssd_net = torch.nn.DataParallel(ssd_net)
        loss_net = torch.nn.DataParallel(loss_net)
        torch.backends.cudnn.benchmark = True

    return {'backbone': ssd_net, 'module': loss_net}


def get_resnet_model(args):
    resnet = ResNet18(args.nClass)
    loss_net = LossNet()

    resnet = resnet.to(args.device)
    loss_net = loss_net.to(args.device)

    if ',' in args.gpu_id:
        resnet = torch.nn.DataParallel(resnet)
        loss_net = torch.nn.DataParallel(loss_net)
        torch.backends.cudnn.benchmark = True

    return {'backbone': resnet, 'module': loss_net}



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
