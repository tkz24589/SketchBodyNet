# import gc
# import os
#
# import cv2
#
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# import torch
# from torchvision.utils import make_grid
# import numpy as np
# import pyrender
# import trimesh
#
# class Renderer:
#     """
#     Renderer used for visualizing the SMPL model
#     Code adapted from https://github.com/vchoutas/smplify-x
#     """
#     def __init__(self, focal_length=5000, img_res=224, faces=None):
#         self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
#                                        viewport_height=img_res,
#                                        point_size=1.0)
#         self.focal_length = focal_length
#         self.camera_center = [img_res // 2, img_res // 2]
#         self.faces = faces
#         self.scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
#         self.material = pyrender.MetallicRoughnessMaterial(
#             metallicFactor=0.2,
#             alphaMode='OPAQUE',
#             baseColorFactor=(0.8, 0.3, 0.3, 1.0))
#
#     def visualize_tb(self, vertices, camera_translation, images):
#         vertices = vertices.cpu().detach().numpy()
#         camera_translation = camera_translation.cpu().detach().numpy()
#         images = images.cpu()
#         images_np = np.transpose(images.numpy(), (0,2,3,1))
#         rend_imgs = []
#         for i in range(vertices.shape[0]):
#             rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
#             rend_imgs.append(images[i])
#             rend_imgs.append(rend_img)
#         rend_imgs = make_grid(rend_imgs, nrow=2)
#         return rend_imgs
#
#     def get_mask(self, vertices, camera_translation, images):
#         vertices = vertices.cpu().detach().numpy()
#         camera_translation = camera_translation.cpu().detach().numpy()
#         images = images.cpu()
#         images_np = np.transpose(images.numpy(), (0,2,3,1))
#         rend_imgs = []
#         # 内存爆炸
#         for i in range(vertices.shape[0]):
#             rend_img = np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))
#             rend_imgs.append(rend_img)
#         rend_imgs = torch.Tensor(np.array(rend_imgs))
#         return rend_imgs
#
#
#     def __call__(self, vertices, camera_translation, image):
#
#         camera_translation[0] *= -1.
#
#         mesh = trimesh.Trimesh(vertices, self.faces)
#         rot = trimesh.transformations.rotation_matrix(
#             np.radians(180), [1, 0, 0])
#         mesh.apply_transform(rot)
#         # mesh.export("result.obj")
#         # mesh.show()
#         mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)
#
#         # self.scene = pyrender.self.scene(ambient_light=(0.5, 0.5, 0.5))
#         self.scene.add(mesh, 'mesh')
#
#         camera_pose = np.eye(4)
#         camera_pose[:3, 3] = camera_translation
#         # camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
#         #                                    cx=self.camera_center[0], cy=self.camera_center[1])
#         camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
#                                            cx=image.shape[1] // 2, cy=image.shape[0] // 2)
#         self.scene.add(camera, pose=camera_pose)
#
#
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
#         light_pose = np.eye(4)
#
#         light_pose[:3, 3] = np.array([0, -1, 1])
#         self.scene.add(light, pose=light_pose)
#
#         light_pose[:3, 3] = np.array([0, 1, 1])
#         self.scene.add(light, pose=light_pose)
#
#         light_pose[:3, 3] = np.array([1, 1, 2])
#         self.scene.add(light, pose=light_pose)
#
#         #修改
#         self.renderer.__setattr__('viewport_width', image.shape[1])
#         self.renderer.__setattr__('viewport_height', image.shape[0])
#
#         color, rend_depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
#         color = color.astype(np.float32) / 255.0
#         valid_mask = (rend_depth > 0)[:,:,None]
#         # output_img = (color[:, :, :3] * valid_mask +
#         #           (1 - valid_mask) * image)
#         output_img = color[:, :, :3] * valid_mask
#         self.scene.clear()
#         del mesh
#         gc.collect()
#         # cv2.imshow('premask', output_img*255)
#         # cv2.waitKey(0)
#         return output_img
"""
Parts of the code are taken from from https://github.com/akanazawa/hmr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight

# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple':(118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = (0, 0, 255)
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, black, 2)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


def visualize_reconstruction(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)
    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton(img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'pink': [.5, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r
