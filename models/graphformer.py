import torch
import torch.nn as nn

from models.feature_net import get_senet, get_resnet50
from models.graph_layers import GraphLinear, GraphResBlock
from models.transformer import Transformer
from utils import Mesh


class GraphFormer(nn.Module):
    def __init__(self, A, template_vertices, num_layers=5, num_channels=256):
        super(GraphFormer, self).__init__()

        self.A = A
        # t, s =  A._indices().detach().cpu().numpy(), A._values().detach().cpu().numpy()
        self.template_vertices = template_vertices
        self.feature_net = get_resnet50(pretrained=True)
        layers = [GraphLinear(3 + 2048, 2 * num_channels), GraphResBlock(2 * num_channels, num_channels, A)]
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                       nn.ReLU(inplace=True),
                                       GraphLinear(num_channels, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(A.shape[0], 3))
        # self.vfomer = Transformer(dim=template_vertices.shape[1], depth=1, heads=6, dim_head=16, mlp_dim=2048)

    def forward(self, image):
        batch_size = image.shape[0]
        template_vertices = self.template_vertices[None, :, :].expand(batch_size, -1, -1)
        feature = self.feature_net(image)
        image_enc = feature.view(batch_size, 2048, 1).expand(-1, -1, template_vertices.shape[-1])
        x = torch.cat([template_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        # shape = self.vfomer(shape)
        return shape, camera


def graphformer(mesh):
    model = GraphFormer(mesh.adjmat, mesh.ref_vertices.t())
    return model
