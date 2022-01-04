'''
Reference:
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

# Loss Prediction Network
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512],
                 interm_dim=128,
                 task='clf'):
        super(LossNet, self).__init__()

        self.GAP = []
        if task == 'clf':
            for feature_size in feature_sizes:
                self.GAP.append(nn.AvgPool2d(feature_size))

        elif task == 'detection':
            for feature_size in feature_sizes:
                self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.FC = []
        for num_channel in num_channels:
            self.FC.append(nn.Linear(num_channel, interm_dim))

        self.GAP = nn.ModuleList(self.GAP)
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, 1)

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out