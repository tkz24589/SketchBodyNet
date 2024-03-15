# # encoding:utf-8
# # Modify from torchvision
# # ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import init
#
# import config
# from models.feature_net import get_resnet50, get_senet
# from models.smpl import SMPL
# from models.transformer import Transformer
# from utils.geometry import rot6d_to_rotmat
#
# """
# This file contains the definitions of the various ResNet models.
# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
# Forward pass was modified to discard the last fully connected layer
# """
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # remove final fully connected layer
#         # x = self.fc(x)
#
#         return x
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
# class HPR(nn.Module):
#     def __init__(self, smpl_mean_params=None):
#         super(HPR, self).__init__()
#         self.cam_size = 3
#         self.pose_size = 144
#         self.shape_size = 10
#         self.smpl = SMPL(config.SMPL_MODEL_DIR,
#                          create_transl=False)
#         self.feature_net = get_senet()
#         self._init_params()
#         self.pose_tranformer = Transformer(dim=2048 + self.pose_size, depth=3, heads=2,
#                                            dim_head=8, mlp_dim=3072, out_dim=self.pose_size)
#
#         self.shape_tranformer = Transformer(dim=2048 + self.shape_size, depth=3, heads=2,
#                                             dim_head=8, mlp_dim=3072, out_dim=self.shape_size)
#
#         self.cam_tranformer = Transformer(dim=2048 + self.cam_size, depth=3, heads=2,
#                                           dim_head=8, mlp_dim=3072, out_dim=self.cam_size)
#
#         mean_params = np.load(smpl_mean_params)
#         init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
#         init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
#         init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
#         self.register_buffer('init_pose', init_pose)
#         self.register_buffer('init_shape', init_shape)
#         self.register_buffer('init_cam', init_cam)
#
#     def _init_params(self):
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 init.kaiming_uniform_(module.weight)
#                 if module.bias is not None:
#                     init.constant_(module.bias, 0)
#
#     def forward(self, x, init_pose=None, init_shape=None, init_cam=None):
#         """
#         :param image:
#         :return:
#         """
#         batch_size = x.shape[0]
#
#         if init_pose is None:
#             init_pose = self.init_pose.expand(batch_size, 1, -1)
#         if init_shape is None:
#             init_shape = self.init_shape.expand(batch_size, 1, -1)
#         if init_cam is None:
#             init_cam = self.init_cam.expand(batch_size, 1, -1)
#
#         feature = self.feature_net(x)
#         pred_pose = self.pose_tranformer(torch.cat([feature, init_pose], 2)).squeeze(1)
#         pred_shape = self.shape_tranformer(torch.cat([feature, init_shape], 2)).squeeze(1)
#         pred_cam = self.cam_tranformer(torch.cat([feature, init_cam], 2)).squeeze(1)
#         pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
#         return pred_pose, pred_shape, pred_cam
#
#
# def hpr(smpl_mean_params=None,
#         pretrained=False,
#         checkpoint_file=None):
#     """ Constructs an HMR model with ResNet50 backbone.
#         Args:
#             pretrained (bool): If True, returns a model pre-trained on ImageNet
#             :param checkpoint_file:
#             :param smpl_mean_params:
#         """
#     model = HPR(smpl_mean_params)
#     if pretrained:
#         checkpoint = torch.load(checkpoint_file)
#         model.load_state_dict(checkpoint['model'], strict=False)
#     return model
# encoding:utf-8
# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import init

import config
import constants
from models.feature_net import get_resnet50, get_senet
from models.transformer import Transformer
from utils.geometry import rot6d_to_rotmat

"""
This file contains the definitions of the various ResNet models.
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
Forward pass was modified to discard the last fully connected layer
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        # remove final fully connected layer
        # x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class HPR(nn.Module):
    def __init__(self, smpl_mean_params=None, type='all'):
        super(HPR, self).__init__()
        self.cam_size = 3
        self.pose_size = 144
        self.shape_size = 10
        self.type = type
        self.feature_net = resnet50(pretrained=True)

        self.pose_regressor = nn.Sequential(
            nn.Linear(2048 + self.pose_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, self.pose_size)
        )
        self.shape_regressor = nn.Sequential(
            nn.Linear(2048 + self.shape_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, self.shape_size)
        )
        self.cam_regressor = nn.Sequential(
            nn.Linear(2048 + self.cam_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, self.cam_size)
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None):
        """
        :param image:
        :return:
        """
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        feature = self.feature_net(x)

        x1 = feature
        if self.type == '3d':
            with torch.no_grad():
                x2 = feature
                x3 = feature
        else:
            x2 = feature
            x3 = feature

        pred_pose = self.pose_regressor(torch.cat([x1, init_pose], dim=1))
        if self.type == '3d':
            with torch.no_grad():
                pred_shape = self.shape_regressor(torch.cat([x2, init_shape], dim=1))
                pred_cam = self.cam_regressor(torch.cat([x3, init_cam], dim=1))
        else:
            pred_shape = self.shape_regressor(torch.cat([x2, init_shape], dim=1))
            pred_cam = self.cam_regressor(torch.cat([x3, init_cam], dim=1))

        pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_pose, pred_shape, pred_cam


def hpr_no_attention_net(smpl_mean_params=None,
        pretrained=False,
        checkpoint_file=None,
        type='all'):
    """ Constructs an HMR model with ResNet50 backbone.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            :param checkpoint_file:
            :param smpl_mean_params:
        """
    model = HPR(smpl_mean_params, type)
    if pretrained:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model

if __name__== "__main__" :
    net = hpr(smpl_mean_params=config.SMPL_MEAN_PARAMS)
    images = torch.ones((64, 3, 224, 224))
    net(images)