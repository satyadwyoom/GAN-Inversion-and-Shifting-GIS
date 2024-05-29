import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np
from typing import Any, Dict, Optional
import torch.nn.utils.spectral_norm as spectral_norm


from torch import Tensor


def save_hook(module, input, output):
    setattr(module, 'output', output)


class LatentShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictor, self).__init__()
        self.features_extractor = resnet18()
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()

class LatentShiftPredictorv2(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictorv2, self).__init__()
        self.features_extractor = resnet18()
        self.features_extractor.conv1 = nn.Conv2d(
            3, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))

    def forward(self, x1):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1 = F.interpolate(x1, self.downsample)
        self.features_extractor(x1)
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)

        return logits


class LatentShiftPredictorV3(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictorV3, self).__init__()
        self.features_extractor = resnet18()
        self.features_extractor.conv1 = nn.Conv2d(
            3, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 512)

    def forward(self, x1):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1 = F.interpolate(x1, self.downsample)
        self.features_extractor(x1)
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift

# class LatentReconstructor(nn.Module):

#     def __init__(self, y_dim) -> None:
#         super(LatentReconstructor, self).__init__()
        
#         self.y_dim = y_dim
#         self.conv1 = nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=False)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#             # State size. 64 x 64 x 64
#         self.conv2 = nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)
#         self.bn1 = nn.BatchNorm2d(128)
#             # nn.LeakyReLU(0.2, True),
#             # State size. 128 x 32 x 32
#         self.conv3 = nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)
#         self.bn2 = nn.BatchNorm2d(256)
#             # nn.LeakyReLU(0.2, True),
#             # State size. 256 x 16 x 16
#         # self.conv4 = nn.Conv2d(256+y_dim, 512, (4, 4), (2, 2), (1, 1), bias=False) ### Comment for 64x64
#         self.bn3 = nn.BatchNorm2d(256)
#             # nn.LeakyReLU(0.2, True),

#         self.fc1_final = nn.Linear(256*8*8, 1024)
#         self.fc2_final = nn.Linear(1024, 1)
#         self.sigmoid_func = nn.Sigmoid()

#     def conv_cond_cat(self, x, y):
#         y_dummy = y[:,:, None, None] * torch.ones(x.shape[0], y.shape[1], x.shape[2], x.shape[3]).to(self.device, dtype=torch.float)
#         x_cat = torch.cat([x,y_dummy], 1)
#         return x_cat.float()


#     def forward(self, x: Tensor, y: Tensor, device='cpu') -> Tensor:
#         # out = self.main(x)
#         self.device = device
#         in_cat = x
#         out = self.lrelu(self.conv1(in_cat))
#         out = self.lrelu(self.bn1(self.conv2(out)))
#         out = self.lrelu(self.bn2(self.conv3(out)))
#         out = self.lrelu(self.bn3(self.conv4(out)))
#         out = torch.flatten(out, 1)

#         out = self.fc1_final(out)
#         out = self.fc2_final(out)
#         # out = self.sigmoid_func(out)

#         return out


class LatentReconstructor(nn.Module):
    def __init__(self, batch_size):
        super(LatentReconstructor, self).__init__()

        param = {'n_channels':3,
                 'D_h_size': 32,
                 'image_size':128,
                 'num_outcomes':512,
                 'use_adaptive_reparam': True,
                 'batch_size': batch_size }
        self.param = param
        model = []

        # start block
        model.append(spectral_norm(nn.Conv2d(self.param['n_channels'], self.param['D_h_size'], kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = self.param['image_size'] // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(self.param['D_h_size'] * mult, self.param['D_h_size'] * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(param['D_h_size'] * mult * 4 * 4)
        out_size = self.param['num_outcomes'] * self.param['batch_size']
        self.fc = spectral_norm(nn.Linear(in_size, out_size, bias=False))

        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))

    def forward(self, input):
        y = self.model(input)
        y = y.view(-1, self.param['D_h_size'] * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.param['num_outcomes'])


        # re-parameterization trick
        if self.param['use_adaptive_reparam']:
            stat_tuple = self.reparam(y)
            stat_tuple = stat_tuple.view(-1, self.param['num_outcomes'] * 2).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(self.param['batch_size'], self.param['num_outcomes'], 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.param['num_outcomes'])

        return output

class LeNetShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetShiftPredictor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.convnet(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()
