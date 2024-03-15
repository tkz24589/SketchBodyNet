"""
    SENet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

import os
import math

import cv2
import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np

import config
from utils.mesh import Mesh
from utils.part_utils import PartRenderer
from .graph_cnn import GraphCNN
from .sketch_trans3D import sketchTrans3D
from .smpl import SMPL
from models.common import conv1x1_block, conv3x3_block, SEBlock
from utils.geometry import rot6d_to_rotmat


class SENetBottleneck(nn.Module):
    """
    SENet bottleneck block for residual path in SENet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width):
        super(SENetBottleneck, self).__init__()
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        group_width2 = group_width // 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=group_width2)
        self.conv2 = conv3x3_block(
            in_channels=group_width2,
            out_channels=group_width,
            stride=stride,
            groups=cardinality)
        self.conv3 = conv1x1_block(
            in_channels=group_width,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SENetUnit(nn.Module):
    """
    SENet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    identity_conv3x3 : bool, default False
        Whether to use 3x3 convolution in the identity link.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width,
                 identity_conv3x3):
        super(SENetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = SENetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)
        self.se = SEBlock(channels=out_channels)
        if self.resize_identity:
            if identity_conv3x3:
                self.identity_conv = conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None)
            else:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEInitBlock(nn.Module):
    """
    SENet specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(SEInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


class SENet(nn.Module):
    """
    SENet model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000,
                 smpl_mean_params=None,
                 ):
        super(SENet, self).__init__()
        npose = 24 * 6
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", SEInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            identity_conv3x3 = (i != 0)
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SENetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    identity_conv3x3=identity_conv3x3))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        # self.output = nn.Sequential()
        # self.output.add_module("dropout", nn.Dropout(p=0.2))
        # self.output.add_module("fc", nn.Linear(
        #     in_features=in_channels,
        #     out_features=num_classes))

        self._init_params()

        self.fc1 = nn.Linear(512 * bottleneck_width + npose + 13, 1024)

        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        self.decshape = nn.Linear(1024, 10)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        x = self.features(x)
        x = x.view(x.size(0), -1)

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            with torch.no_grad():
                pred_cam = self.deccam(xc) + pred_cam

        # pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_pose, pred_shape, pred_cam, x


def get_senet(blocks,
              checkpoint_file,
              pretrained=False,
              smpl_mean_params=None):
    """
    Create SENet model with specific parameters.
    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 16:
        layers = [1, 1, 1, 1]
        cardinality = 32
    elif blocks == 28:
        layers = [2, 2, 2, 2]
        cardinality = 32
    elif blocks == 40:
        layers = [3, 3, 3, 3]
        cardinality = 32
    elif blocks == 52:
        layers = [3, 4, 6, 3]
        cardinality = 32
    elif blocks == 103:
        layers = [3, 4, 23, 3]
        cardinality = 32
    elif blocks == 154:
        layers = [3, 8, 36, 3]
        cardinality = 64
    else:
        raise ValueError("Unsupported SENet with number of blocks: {}".format(blocks))

    bottleneck_width = 4
    init_block_channels = 128
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SENet(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        smpl_mean_params=smpl_mean_params)

    if pretrained and checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        try:
            net.load_state_dict(checkpoint['model'], strict=False)
        except:
            net.load_state_dict(checkpoint['shpr'], strict=False)
        print('cnn Checkpoint loaded')

    return net

class SHPR(nn.Module):
    def __init__(self, mesh, num_layers, num_channels, smpl_mean_params, pretrained, checkpoint_file):
        super(SHPR, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         create_transl=False)
        self.mesh = mesh
        self.ref_vertices = mesh.ref_vertices.t()
        self.feature = get_senet(blocks=16,
                                 smpl_mean_params=smpl_mean_params,
                                 pretrained=pretrained,
                                 checkpoint_file=checkpoint_file)
        self.gnn = GraphCNN(mesh.adjmat, num_layers, num_channels)

    def forward(self, image):
        """
        :param image:
        :return:
        """
        ref_vertices = self.ref_vertices[None, :, :].expand(image.shape[0], -1, -1)
        pred_pose, pred_betas, pred_camera, features = self.feature(image)
        pred_vertices_cnn = self.smpl(pred_pose, pred_betas).vertices
        pred_vertices_sub_cnn = self.mesh.downsample(pred_vertices_cnn)
        pred_vertices_sub = self.gnn(features, pred_vertices_sub_cnn.transpose(1, 2))
        pred_vertices_tran = self.mesh.upsample(pred_vertices_sub.transpose(1, 2))

        return pred_pose, pred_betas, pred_camera, pred_vertices_tran, pred_vertices_cnn


def shpr(smpl_mean_params=None,
         pretrained=True,
         model_name='shape',
         pose_checkpoint_file=None):
    """ Constructs an HMR model with ResNet50 backbone.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            :param num_channels:
            :param num_layers:
            :param mesh:
            :param checkpoint_file:
            :param smpl_mean_params:
            :param pretrained:
            :param model_name:
        """
    if model_name == 'pose':
        model = get_senet(blocks=16,
                          smpl_mean_params=smpl_mean_params,
                          pretrained=pretrained,
                          checkpoint_file=pose_checkpoint_file)
    if model_name == 'shape':
        mesh = Mesh()
        model = SHPR(mesh=mesh,
                     num_layers=5,
                     num_channels=256,
                     smpl_mean_params=smpl_mean_params,
                     pretrained=pretrained,
                     checkpoint_file=pose_checkpoint_file)
    return model
