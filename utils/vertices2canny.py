import cv2
import numpy as np
import trimesh

import config
import constants
from renderer.visualizer import Visualizer
from utils import imutils
from utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from utils.imutils import transform, flip_kp


class V2C:
    def __init__(self, device):
        from models.smpl import SMPL
        self.smpl = SMPL(config.SMPL_MODEL_DIR).to(device)
        self.visualizer = Visualizer('opengl')

    def go(self, vertices, org_img, center, scale, camera, img_path):
        img, boxScale_o2n, bboxTopLeft = imutils.crop_bboxInfo(org_img, center, scale, (constants.IMG_RES, constants.IMG_RES))
        pred_vertices_bbox = convert_smpl_to_bbox(vertices[0].cpu().detach().numpy(), camera[0],
                                                  camera[1:])
        pred_vertices_img = convert_bbox_to_oriIm(
            pred_vertices_bbox, boxScale_o2n, bboxTopLeft, org_img.shape[1], org_img.shape[0])

        # 导出mesh_2d
        out_mesh = trimesh.Trimesh(pred_vertices_img, self.smpl.faces, process=False)
        rot_mesh = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot_mesh)
        pred_mesh_list = [{'vertices': pred_vertices_img,
                           'faces': self.smpl.faces}]
        res_img = self.visualizer.visualize(
            input_img=org_img,
            pred_mesh_list=pred_mesh_list,
            mesh_2d=True)
        edges = cv2.Canny(res_img, 100, 200)
        edges = cv2.resize(edges, (2048, 2048))
        cv2.imwrite(img_path, edges)
        cv2.imshow('canny', edges)
        cv2.waitKey(0)
def augm_params():
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling

    return flip, pn, rot, sc


def j2d_processing(kp, center, scale, r, f):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                               [constants.IMG_RES, constants.IMG_RES], rot=r)
    # convert to normalized coordinates
    kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
    # flip the x coordinates
    if f:
        kp = flip_kp(kp)
    kp = kp.astype('float32')
    return kp
