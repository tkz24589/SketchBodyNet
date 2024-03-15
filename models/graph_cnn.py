"""
This file contains the Definition of GraphCNN
"""
from __future__ import division

import torch
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear


class GraphCNN(nn.Module):

    def __init__(self, A, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        layers = [GraphLinear(3 + 2048, 2 * num_channels), GraphResBlock(2 * num_channels, num_channels, A)]
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)

    def forward(self, feature, ref_vertices_smpl):
        """Forward pass
        Inputs:
            feature: (B, 2048)
            ref_vertices_smpl: (B, 3, 1723)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
        """
        batch_size = feature.shape[0]
        image_enc = feature.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices_smpl.shape[-1])
        x = torch.cat([ref_vertices_smpl, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        return shape
