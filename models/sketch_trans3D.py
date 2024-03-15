"""
This file contains the Definition of GraphCNN
"""
from __future__ import division

import torch
import torch.nn as nn


class sketchTrans3D(nn.Module):

    def __init__(self):
        super(sketchTrans3D, self).__init__()
        self.li

    def forward(self, image, feature, ref_vertices_smpl):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
            feature: (B, 2048)
            ref_vertices_smpl: (B, 1723, 3)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
        """
        batch_size = image.shape[0]

        return shape
