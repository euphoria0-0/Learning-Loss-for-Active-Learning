from torch.nn import init

from .lossnet import *
from .ssd import *
from data.voc_data import voc_cfg



def get_model(args, phase='train'):
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

    loss_net = LossNet(num_channels=[512, 1024, 512, 256, 256, 256])

    return {'backbone': ssd_net, 'module': loss_net}



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
