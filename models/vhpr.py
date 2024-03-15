# import torch
# from torch import nn
# import numpy as np
# import constants
# from utils.geometry import rot6d_to_rotmat
#
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
#
# import torch.nn as nn
#
# # helpers
#
# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)
#
#
# # classes
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
#
#
# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='mean', channels=3,
#                  dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         # self.mlp_head = nn.Sequential(
#         #     nn.LayerNorm(dim),
#         #     nn.Linear(dim, num_classes)
#         # )
#
#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#
#         x = self.transformer(x)
#
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return x
#
#
# class VHPR(nn.Module):
#     """ SMPL Iterative Regressor with ResNet50 backbone
#     """
#
#     def __init__(self, smpl_mean_params):
#         self.inplanes = 64
#         super(VHPR, self).__init__()
#         self.pose_size = 24 * 6
#         self.shape_size = 10
#         self.cam_size = 3
#
#         self.net = ViT(
#             image_size=constants.IMG_RES,
#             patch_size=32,
#             dim=256,
#             depth=6,
#             heads=16,
#             mlp_dim=2048,
#             dropout=0.1,
#             emb_dropout=0.1
#         )
#         self.tranpose = ViT(
#             image_size=self.pose_size + 256,
#             patch_size=40,
#             dim=self.pose_size,
#             depth=3,
#             heads=8,
#             mlp_dim=512,
#             dropout=0.,
#             emb_dropout=0.
#         )
#         self.transhape = ViT(
#             image_size=self.shape_size + 256,
#             patch_size=38,
#             dim=self.shape_size,
#             depth=3,
#             heads=8,
#             mlp_dim=256,
#             dropout=0.,
#             emb_dropout=0.
#         )
#         self.trancam = ViT(
#             image_size=self.cam_size + 256,
#             patch_size=37,
#             dim=self.cam_size,
#             depth=3,
#             heads=8,
#             mlp_dim=128,
#             dropout=0.,
#             emb_dropout=0.
#         )
#
#         mean_params = np.load(smpl_mean_params)
#         init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
#         init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
#         init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
#         self.register_buffer('init_pose', init_pose)
#         self.register_buffer('init_shape', init_shape)
#         self.register_buffer('init_cam', init_cam)
#
#     def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
#
#         batch_size = x.shape[0]
#
#         if init_pose is None:
#             init_pose = self.init_pose.expand(batch_size, -1)
#         if init_shape is None:
#             init_shape = self.init_shape.expand(batch_size, -1)
#         if init_cam is None:
#             init_cam = self.init_cam.expand(batch_size, -1)
#
#         out = self.net(x)
#
#         pred_pose = init_pose
#         pred_shape = init_shape
#         pred_cam = init_cam
#
#         for _ in range(n_iter):
#             xp = torch.cat([pred_pose, out], 1).view(batch_size, 1, 256 + self.pose_size, 1).expand(-1, 3, -1, 256 + self.pose_size)
#             xs = torch.cat([pred_shape, out], 1).view(batch_size, 1, 256 + self.shape_size, 1).expand(-1, 3, -1, 256 + self.shape_size)
#             xc = torch.cat([pred_cam, out], 1).view(batch_size, 1, 256 + self.cam_size, 1).expand(-1, 3, -1, 256 + self.cam_size)
#             pred_pose = self.tranpose(xp)
#             pred_shape = self.transhape(xs)
#             pred_cam = self.trancam(xc)
#
#         pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
#
#         return pred_rotmat, pred_shape, pred_cam
#
#
# def vhpr(smpl_mean_params):
#     """ Constructs an HMR model with ResNet50 backbone.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VHPR(smpl_mean_params)
#     return model
import torch
from torch import nn
import numpy as np

import config
import constants
from utils.geometry import rot6d_to_rotmat

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn as nn

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='mean', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class VHPR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, smpl_mean_params):
        self.inplanes = 64
        super(VHPR, self).__init__()
        self.pose_size = 24 * 6
        self.shape_size = 10
        self.cam_size = 3

        self.net = ViT(
            image_size=constants.IMG_RES,
            patch_size=28,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.fc1 = nn.Linear(1024 + self.pose_size + self.shape_size + self.cam_size, 1024)
        self.depose = nn.Linear(1024, self.pose_size)
        self.deshape = nn.Linear(1024, self.shape_size)
        self.decam = nn.Linear(1024, self.cam_size)
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        out = self.net(x)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for _ in range(n_iter):
            out = self.fc1(torch.cat([out, pred_pose, pred_shape, pred_cam], dim=1))
            pred_pose = self.depose(out)
            pred_shape = self.deshape(out)
            pred_cam = self.decam(out)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


def vhpr(smpl_mean_params):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VHPR(smpl_mean_params)
    return model

if __name__=='__main__':
    net = vhpr(config.SMPL_MEAN_PARAMS)
    images = torch.ones((32, 3, 224, 224))
    out = net(images)
