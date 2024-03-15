import os

import numpy as np
from tqdm import tqdm

from models.shmr import shmr
from models.thpr import thpr
from models.vhpr import vhpr
from utils import TrainOptions
import constants
import torch
from torchgeometry import angle_axis_to_rotation_matrix
import cv2
from models import SMPL, hmr
from models.thmr import thmr
import config
import trimesh

from utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

from utils.geometry import estimate_translation
import utils.imutils as imutils
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

from renderer.visualizer import Visualizer
# from utils.part_utils import PartRenderer
from PIL.Image import Image

import utils.geometry_utils as gu
from torchvision.transforms import Normalize

from utils.part_utils import PartRenderer


def skech_to_mask(img, SavePath):
    # 复制 im_in 图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_floodfill = img.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = img | im_floodfill_inv

    # 保存结果
    cv2.imwrite(SavePath, im_out)
    return SavePath

def augm_params(options, is_train):
    """Get augmentation parameters."""

    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if is_train:
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - options.noise_factor, 1 + options.noise_factor, 3)

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(2 * options.rot_factor,
                  max(-2 * options.rot_factor, np.random.randn() * options.rot_factor))

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + options.scale_factor,
                 max(1 - options.scale_factor, np.random.randn() * options.scale_factor + 1))

        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    return flip, pn, rot, sc


def flip_pose(pose, is_flipped, flipped_parts):
    """flip SMPL pose parameters"""
    is_flipped = is_flipped.byte()
    pose_f = pose.clone()
    pose_f[is_flipped, :] = pose[is_flipped][:, flipped_parts]
    # we also negate the second and the third dimension of the axis-angle representation
    pose_f[is_flipped, 1::3] *= -1
    pose_f[is_flipped, 2::3] *= -1
    return pose_f


def rotate_pose(pose, rot):
    """Rotate SMPL pose parameters by rot degrees"""
    pose = pose.clone()
    cos = torch.cos(-np.pi * rot / 180.)
    sin = torch.sin(-np.pi * rot / 180.)
    zeros = torch.zeros_like(cos)
    r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
    r3[:, 0, -1] = 1
    R = torch.cat([torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
                   torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1),
                   r3], dim=1)
    global_pose = pose[:, :3]
    global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
    global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
    global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3)
    global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
    global_pose_rotmat = global_pose_rotmat[:, :-1, :-1].cpu().numpy()
    global_pose_np = np.zeros((global_pose.shape[0], 3))
    for i in range(global_pose.shape[0]):
        aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
        global_pose_np[i, :] = aa.squeeze()
    pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
    return pose


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


def rgb_processing(rgb_img, center, scale, rot, flip, pn):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale,
                   [constants.IMG_RES, constants.IMG_RES], rot=rot)
    # cv2.imshow('img', rgb_img)
    # cv2.waitKey(0)
    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    return rgb_img

def test():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    options = TrainOptions().parse_args()

    # 计算obj_2d smplify
    model = shmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load('logs/resume_check/train_example/checkpoints/2022_09_02-12_09_53_loss_1.7051371513756677.pt')
    model.load_state_dict(checkpoint['model'], strict=False)
    # self.model_regressor.load_state_dict(checkpoint, strict=False)
    model.eval()
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
               create_transl=False).to(device)
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    # visualizer = Visualizer('opengl')
    path_root = 'datasets/data/mpi_inf_3dhp/obj_canny/'
    test_file = np.load('data/dataset_extras/mpi-inf-3dhp_train_sketch.npz')
    imgname = test_file['imgname']
    center = test_file['center']
    scale = test_file['scale']
    renderer = PartRenderer()
    with torch.no_grad():
        rotmat, betas, camera = [], [], []
        # for i in range(len(imgname)):
        for i in tqdm(range(len(imgname)), total=len(imgname)):
            org_img = cv2.imread(path_root + imgname[i])
            img = cv2.imread(path_root + imgname[i])[:, :, ::-1].copy().astype(np.float32)
            # img = cv2.resize(img, (2048, 2048))
            flip, pn, rot, sc = augm_params(options, False)
            img = rgb_processing(img, center[i], sc * scale[i], rot, flip, pn)
            cv2.imshow('b', img[0])
            img = torch.from_numpy(img).float()
            # Store image before normalization to use it in visualization
            img = normalize_img(img).reshape((1, 3, 224, 224)).to(device)

            # 回归smpl参数
            pred_rotmat, pred_betas, pred_camera = model(img)
            # rotmat.append(pred_rotmat.detach().cpu())
            # betas.append(pred_betas.detach().cpu())
            # camera.append(pred_camera.detach().cpu())
            pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
            pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
            body_pose = pred_aa[:, 3:]
            global_orient = pred_aa[:, :3]

            # 得到mesh
            opt_output = smpl(betas=pred_betas, body_pose=body_pose,
                              global_orient=global_orient, pose2rot=False)

            pre_mask, _ = renderer(opt_output.vertices, pred_camera, False)
            cv2.imshow('r', pre_mask[0].cpu().numpy())
            cv2.waitKey(0)
            # camera_t = pred_camera.cpu().numpy().ravel()
            #
            # img, boxScale_o2n, bboxTopLeft = imutils.crop_bboxInfo(org_img,
            #                                                        center[i], scale[i], (224, 224))
            # pred_vertices_bbox = convert_smpl_to_bbox(opt_output.vertices[0].cpu().detach().numpy(), camera_t[0], camera_t[1:])
            # pred_vertices_img = convert_bbox_to_oriIm(
            #     pred_vertices_bbox, boxScale_o2n, bboxTopLeft, org_img.shape[1], org_img.shape[0])
            #
            # # 导出mesh_2d
            # out_mesh = trimesh.Trimesh(pred_vertices_img, smpl.faces, process=False)
            # rot_mesh = trimesh.transformations.rotation_matrix(
            #     np.radians(180), [1, 0, 0])
            # out_mesh.apply_transform(rot_mesh)
            # pred_mesh_list = [{'vertices': pred_vertices_img,
            #                    'faces': smpl.faces}]
            # res_img = visualizer.visualize(
            #     255-org_img,
            #     pred_mesh_list=pred_mesh_list,
            #     mesh_2d=True)
    # np.savez('lsp-orig_train_y.npz',
    #          rotmat=rotmat,
    #          betas=betas,
    #          camera=camera
    #          )
            # cv2.imwrite('examples/' + imgname[i].split('/')[1] + imgname[i].split('/')[-1], res_img)
            # cv2.imwrite('examples/' + 'org_' + imgname[i].split('/')[1] + imgname[i].split('/')[-1], 255-org_img)
            # cv2.imwrite('examples/' + imgname[i].split('/')[-1], res_img)
            # if isinstance(res_img, Image):
            #     # inputImg = cv2.cvtColor(np.array(inputImg), cv2.COLOR_RGB2BGR)
            #     res_img = np.array(res_img)
            # edges = cv2.Canny(res_img, 100, 200)
            # cv2.imwrite('datasets/data/lsp-orig/obj_2d_img/' + imgname[i].split('/')[-1], res_img)
            # cv2.imwrite('datasets/data/lsp-orig/obj_canny/' + imgname[i].split('/')[-1], edges)
            # out_mesh.export("/home/stu/2Dto3D/pose2smplx/frankmocap-main/sample_data/obj/result.obj")
            # cv2.imshow("result", res_img)
            # cv2.waitKey(0)
            # out_mesh.show()
def obj2img():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    options = TrainOptions().parse_args()
    img_root = 'datasets/data/mpi_inf_3dhp/obj_canny/'
    file = np.load('data/dataset_extras/mpi-inf-3dhp_train_sketch.npz')
    center = file['center']
    scale = file['scale']
    imgname = file['imgname']
    # keypoints = np.concatenate([file['openpose'], file['part']], axis=1)
    # keypoints_2d = []
    # for index in range(len(imgname)):
    #     keypoint = keypoints[index].copy()
    #     keypoints_2d.append(j2d_processing(keypoint, center[index], sc[index] * scale[index], rot[index], flip[index]))
    # keypoints_2d = torch.Tensor(keypoints_2d)
    part = file['part']
    file_fit = np.load('data/static_fits/mpi-inf-3dhp_fits.npy')
    flipped_parts = torch.tensor(constants.SMPL_POSE_FLIP_PERM, dtype=torch.int64)
    pose = torch.Tensor(file_fit[:, :72])
    betas = torch.Tensor(file_fit[:, 72:])

    # 计算obj_2d smplify
    model = hmr(config.SMPL_MEAN_PARAMS, pretrained=False).to(device)
    checkpoint = torch.load('logs/resume_check/checkpoints/hmr-2022_06_11-05_51_56_loss_1.3351002931594849.pt')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    # renderer = PartRenderer()
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    visualizer = Visualizer('opengl')

    for i in range(len(imgname)):
        with torch.no_grad():
            img_path = img_root + imgname[i]
            org_img = cv2.imread(img_path)
            img = org_img[:, :, ::-1].copy().astype(np.float32)
            flip, pn, rot, sc = augm_params(options, False)
            img = rgb_processing(img, center[i], sc * scale[i], rot, flip, pn)
            img = torch.from_numpy(img).float()
            # Store image before normalization to use it in visualization
            img = normalize_img(img).reshape((1, 3, 224, 224)).to(device)

            # 回归smpl参数
            pred_rotmat, pred_betas, pred_camera = model(img)
            opt_pose = pose[i].to(device).reshape((1, pose.shape[1]))
            opt_betas = betas[i].to(device).reshape(1, betas.shape[1])
            body_pose = opt_pose[:, 3:]
            global_orient = opt_pose[:, :3]
            # pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
            # pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
            # body_pose = pred_aa[:, 3:]
            # global_orient = pred_aa[:, :3]

            # 得到mesh
            opt_output = smpl(betas=opt_betas, body_pose=body_pose,
                              global_orient=global_orient, pose2rot=False)
            # images_pred = renderer.visualize_tb(opt_output.vertices, pred_camera, img)
            # mask, parts = renderer(opt_output.vertices, pred_camera)
            camera = pred_camera.cpu().numpy().ravel()

            img, boxScale_o2n, bboxTopLeft = imutils.crop_bboxInfo(org_img, center[i], scale[i], (224, 224))
            pred_vertices_bbox = convert_smpl_to_bbox(opt_output.vertices[0].cpu().detach().numpy(), camera[0],
                                                      camera[1:])
            pred_vertices_img = convert_bbox_to_oriIm(
                pred_vertices_bbox, boxScale_o2n, bboxTopLeft, org_img.shape[1], org_img.shape[0])

            # 导出mesh_2d
            out_mesh = trimesh.Trimesh(pred_vertices_img, smpl.faces, process=False)
            rot_mesh = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot_mesh)
            pred_mesh_list = [{'vertices': pred_vertices_img,
                               'faces': smpl.faces}]
            res_img = visualizer.visualize(
                input_img=255 - org_img,
                pred_mesh_list=pred_mesh_list,
                mesh_2d=True)
            skech_to_mask(org_img, 'data/paperdata/mask_' + imgname[i].split('/')[-1])
            cv2.imwrite('data/paperdata/' + imgname[i].split('/')[-1], res_img)
            cv2.imwrite('data/paperdata/real_' + imgname[i].split('/')[-1], 255 - org_img)
            # if isinstance(res_img, Image):
            #     # inputImg = cv2.cvtColor(np.array(inputImg), cv2.COLOR_RGB2BGR)
            #     res_img = np.array(res_img)
            # cv2.imshow('pre_img', res_img)
            # edges = cv2.Canny(res_img, 100, 200)
            # mask = mask.cpu().numpy()
            # part = parts.cpu().numpy()
            # mask = mask[0] * 255
            # part = part[0].astype(float)
            # mask = cv2.resize(mask, (org_img.shape[1], org_img.shape[0]))
            # part = cv2.resize(part, (org_img.shape[1], org_img.shape[0]))
            # mask_path = "data/lsp/" + imgname[i].split('/')[-1].split('.')[0] + '_segmentation.png'
            # part_path = "data/lsp/" + imgname[i].split('/')[-1].split('.')[0] + '_part_segmentation.png'
            # cv2.imwrite(mask_path, mask)
            # cv2.imwrite(part_path, part)
            # edges = cv2.resize(edges, (2048, 2048))
            # cv2.imwrite("datasets/data/mpi_inf_3dhp/obj_canny/" + imgname[i], edges)
            # out_mesh.export("sample_data/obj/" + str(i) + "_result.obj")
            # out_mesh.show()

def resize_dataset():
    file = np.load("data/dataset_extras/mpi_inf_3dhp_valid.npz")
    file_name = file['imgname']
    num = 0
    # test
    for i in range(6):
        S_path = "datasets/data/mpi_inf_3dhp/obj_canny/mpi_inf_3dhp_test_set/" + 'TS' + str(i + 1) + '/'
        img_path = S_path + '/imageSequence/'
        sketch_path_v = S_path + 'sketch_imageSequence/'
        if not os.path.exists(sketch_path_v):
            os.mkdir(sketch_path_v)
        frames = os.listdir(img_path)
        frames.sort(key=lambda x: int(x[6:-4]))
        for f in range(len(frames)):
            path = img_path + frames[f]
            sketch_path = sketch_path_v + frames[f]
            sketch_img = np.zeros((480, 480, 3), np.uint8)
            cv2.imwrite(sketch_path, sketch_img)
            num += 1
    # train
    # for i in range(2):
    #     S_path = "datasets/data/mpi_inf_3dhp/obj_canny/" + 'S' + str(i + 1) + '/'
    #     for j in range(2):
    #         Seq_path = S_path + "Seq" + str(j + 1) + '/imageFrames/'
    #         for k in range(9):
    #             if k == 3:
    #                 continue
    #             frame_path = Seq_path + "video_" + str(k) + '/'
    #             sketch_path_v = Seq_path + "video_sketch_" + str(k) + '/'
    #             if not os.path.exists(sketch_path_v):
    #                 os.mkdir(sketch_path_v)
    #             frames = os.listdir(frame_path)
    #             frames.sort(key=lambda x: int(x[6:-4]))
    #             for f in range(len(frames)):
    #                 path = frame_path + frames[f]
    #                 sketch_path = sketch_path_v + frames[f]
    #                 sketch_img = np.zeros((480, 480, 3), np.uint8)
    #                 cv2.imwrite(sketch_path, sketch_img)
    #                 # image = cv2.imread(path)
    #                 # image_candy = cv2.Canny(image, 100, 200)
    #                 # cv2.imshow("1",image_candy)
    #                 # cv2.waitKey(0)
    #                 exist_path = 'S' + str(i+1) + '/' + "Seq" + str(j + 1) + '/imageFrames/' \
    #                              + "video_" + str(k) + '/' + frames[f]
    #                 if exist_path in file_name:
    #                     num += 1
    #                 else:
    #                     # os.remove("datasets/data/mpi_inf_3dhp/" + exist_path)
    #                     pass
    print(num)
obj2img()
