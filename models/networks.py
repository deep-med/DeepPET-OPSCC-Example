# Include basic network structure of DeepPET-OPSCC
# #


import torch.nn as nn
from collections import OrderedDict

class DCNN_Img_FeatureExtract_Origin(nn.Module):
    """
        DeepPET-OPSCC model
        """
    def __init__(self, channels=[16, 32, 64], kernels=[3, 3, 3], use_lymph=False):
        super(DCNN_Img_FeatureExtract_Origin, self).__init__()


        self.features = nn.Sequential(OrderedDict(

            [
                ('conv11', nn.Conv3d(3 if use_lymph else 2, channels[0], kernel_size=kernels[0], stride=1, padding=1)),
                ('bn11', nn.BatchNorm3d(channels[0])),
                ('relu11', nn.ReLU(inplace=True)),
                ('conv12', nn.Conv3d(channels[0], channels[0], kernel_size=kernels[0], stride=1, padding=1)),
                ('bn12', nn.BatchNorm3d(channels[0])),
                ('relu12', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool3d(kernel_size=2)),

                ('conv21', nn.Conv3d(channels[0], channels[1], kernel_size=kernels[1], stride=1, padding=1)),
                ('bn21', nn.BatchNorm3d(channels[1])),
                ('relu21', nn.ReLU(inplace=True)),
                ('conv22', nn.Conv3d(channels[1], channels[1], kernel_size=kernels[1], stride=1, padding=1)),
                ('bn22', nn.BatchNorm3d(channels[1])),
                ('relu22', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool3d(kernel_size=2)),

                ('conv31', nn.Conv3d(channels[1], channels[2], kernel_size=kernels[2], stride=1, padding=1)),
                ('bn31', nn.BatchNorm3d(channels[2])),
                ('relu31', nn.ReLU(inplace=True)),
                ('conv32', nn.Conv3d(channels[2], channels[2], kernel_size=kernels[2], stride=1, padding=1)),
                ('bn32', nn.BatchNorm3d(channels[2])),
                ('relu32', nn.ReLU(inplace=True))
              ]
        ))

    def forward(self, x):
        out = self.features(x)
        return out



class DeepConvSurv_Cox_Origin(nn.Module):
    """ Cox proportional hazards model use 3DCNN as Image Feature Extractor

    """

    def __init__(self, use_lymph=False):
        super(DeepConvSurv_Cox_Origin, self).__init__()


        self.features = DCNN_Img_FeatureExtract_Origin(use_lymph=use_lymph)

        self.AvgPool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
            )


    def forward(self, x):
        x = self.features(x)

        x = self.AvgPool(x) # (batch, 64, 1, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 64)

        output = self.classifier(x)

        return output
