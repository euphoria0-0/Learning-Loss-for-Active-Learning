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
    target = target.reshape(1) ###
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
    def __init__(self, feature_sizes=[512, 1024, 512, 256, 256, 256],
                 num_channels=[512, 1024, 512, 256, 256, 256],
                 interm_dim=128):
        super(LossNet, self).__init__()

        # self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        # self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        # self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        # self.GAP4 = nn.AvgPool2d(feature_sizes[3])
        # self.GAP5 = nn.AvgPool2d(feature_sizes[4])
        # self.GAP6 = nn.AvgPool2d(feature_sizes[5])
        self.GAP1 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP2 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP3 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP4 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP5 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAP6 = nn.AdaptiveAvgPool2d((1, 1))

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.FC5 = nn.Linear(num_channels[4], interm_dim)
        self.FC6 = nn.Linear(num_channels[5], interm_dim)

        self.linear = nn.Linear(6 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        #print(out1.size()) #[32, 512, 38, 38]
        out1 = out1.view(out1.size(0), -1)
        #print(out1.size()) #[32, 739328]
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out5 = self.GAP5(features[4])
        out5 = out5.view(out5.size(0), -1)
        out5 = F.relu(self.FC5(out5))

        out6 = self.GAP6(features[5])
        out6 = out6.view(out6.size(0), -1)
        out6 = F.relu(self.FC6(out6))

        out = self.linear(torch.cat((out1, out2, out3, out4, out5, out6), 1))
        return out