import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Transda(nn.Module):
    def __init__(self, num_classes, bottleneck_feature=256):
        super().__init__()

        self.feature = timm.create_model('vit_base_r50_s16_224_in21k', pretrained=True)
        self.fn = nn.Sequential(
            nn.Linear(self.feature.head.in_features, bottleneck_feature),
            nn.BatchNorm1d(bottleneck_feature),
        )
        self.cls = nn.Linear(bottleneck_feature, num_classes)
        self.feature.head = EmptyLayer()

    def forward(self, x):
        feature = self.feature(x)
        feature = (self.fn(feature))
        cls = self.cls(feature)

        return feature, cls

if __name__ == '__main__':
    model = Transda(31)
    '''
    for name, param in model.named_modules():
        print(name)
    '''
    print(model)