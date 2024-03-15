import argparse
from os.path import join

import numpy as np
from cv2 import cv2
from tqdm import tqdm

import config
from renderer.visualizer import Visualizer
from utils.bodypose import general_pose_model
from utils.read_openpose import read_openpose

from utils.imutils import crop


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mpi-inf-3dhp', choices=['lsp-orig', 'mpi-inf-3dhp'],
                    help='Choose process dataset')
parser.add_argument('--train', default=False, help='Train or test')


def new_data_file(dataset, is_train):
    if is_train:
        file = np.load(config.DATASET_FILES[1][dataset])
        new_file = join(config.DATASET_NPZ_PATH, dataset + '_train_new.npz')
    else:
        file = np.load(config.DATASET_FILES[0][dataset])
        new_file = join(config.DATASET_NPZ_PATH, dataset + '_valid_new.npz')
    # 更新数据集文件center、scale、openpose
    root_path = ''
    mask_path = 'data/mask/'
    if dataset == 'lsp-orig' and is_train:
        s = ''
        root_path = config.LSP_ROOT
    elif dataset == 'lsp-orig' and not is_train:
        s = ''
        root_path = config.LSP_ROOT
        # masknames = file['maskname']
        partnames = file['partname']
    if dataset == 'mpi-inf-3dhp' and is_train:
        s = '/'
        root_path = config.MPI_INF_3DHP_ROOT
        poses = file['pose']
        shapes = file['shape']
        has_smpls = file['has_smpl']
        Ss = file['S']
    elif dataset == 'mpi-inf-3dhp' and not is_train:
        s = '/sketch_'
        root_path = config.MPI_INF_3DHP_ROOT
        Ss = file['S']
    parts = file['part']
    img_names = file['imgname']
    try:
        openpose = file['openpose']
    except:
        openpose = np.zeros((len(img_names), 25, 3))
    imgnames_, scales_, centers_, openposes_, parts_ = [], [], [], [], []
    Ss_, has_smpls_, shapes_, poses_, masknames_, partnames_ = [], [], [], [], [], []
    for i, img_name in tqdm(enumerate(img_names), total=len(img_names)):
        # path_list = img_name.split('/')
        # imgname = '/'.join(path_list[:-2]) + s + \
        #           path_list[-2] + '/' + path_list[-1]
        imgnames_.append(img_name)
        img = cv2.imread(root_path + '/' + img_name)
        img_ = img[:, :, ::-1].copy()
        box = get_box(img_)
        center, scale = location_from_box(box)
        centers_.append(center)
        scales_.append(scale)
        if is_train and dataset == 'mpi-inf-3dhp':
            img_path = mask_path + img_name.split('/')[0] + '_' + img_name.split('/')[1] + '_' + \
                       img_name.split('/')[-2] + '_' + img_name.split('/')[-1].split('.')[0] + '_real_' + '.png'
        elif not is_train and dataset == 'mpi-inf-3dhp':
            img_path = mask_path + img_name.split('/')[0] + '_' + img_name.split('/')[-3] + '_' + \
                       img_name.split('/')[-1].split('.')[0] + '_real_' + '.png'
        else:
            img_path = mask_path + img_name.split('/')[0] + '_' + img_name.split('/')[-1].split('.')[0] + '_real_' + '.png'
        mask_file = skech_to_mask(img, img_path)
        # openpose, part = pose_from_mask(mask_file, dataset)
        save_mask(img_path, center, scale, (224, 224))
        # openpose = read_openpose(openpose, part, 'mpi_inf_3dhp')
        openposes_.append(openpose[i])
        parts_.append(parts[i])
        masknames_.append(img_path)
        if dataset == 'mpi-inf-3dhp' and is_train:
            Ss_.append(Ss[i])
            has_smpls_.append(has_smpls[i])
            shapes_.append(shapes[i])
            poses_.append(poses[i])
        if dataset == 'mpi-inf-3dhp' and not is_train:
            Ss_.append(Ss[i])
        if dataset == 'lsp-orig' and not is_train:
            partnames_.append(partnames[i])
    if dataset == 'lsp-orig' and is_train:
        np.savez(new_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 part=parts_,
                 openpose=openposes_,
                 maskname=masknames_)
    elif dataset == 'lsp-orig' and not is_train:
        np.savez(new_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 part=parts_,
                 maskname=masknames_,
                 partname=partnames_
                 )
    if dataset == 'mpi-inf-3dhp' and is_train:
        np.savez(new_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 part=parts_,
                 pose=poses_,
                 shape=shapes_,
                 has_smpl=has_smpls_,
                 S=Ss_,
                 openpose=openposes_,
                 maskname=masknames_)
    elif dataset == 'mpi-inf-3dhp' and not is_train:
        np.savez(new_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 part=parts_,
                 S=Ss_,
                 maskname=masknames_)

def process_erro(dataset):
    file = np.load(config.DATASET_FILES[1][dataset])
    new_file = 'data/dataset_extras/mpi-inf-3dhp_train_sketch.npz'
    # 更新数据集文件center、scale、openpose
    root_path = ''
    mask_path = 'data/mask/'
    if dataset == 'lsp-orig':
        root_path = config.LSP_ROOT
    if dataset == 'mpi-inf-3dhp':
        root_path = config.MPI_INF_3DHP_ROOT
    centers = file['center']
    scales = file['scale']
    poses = file['pose']
    parts = file['part']
    shapes = file['shape']
    has_smpls = file['has_smpl']
    Ss = file['S']
    img_names = file['imgname']
    openposes = file['openpose']

    imgnames_, scales_, centers_, openposes_, parts_ = [], [], [], [], []
    Ss_, has_smpls_, shapes_, poses_ = [], [], [], []
    for i, img_name in enumerate(img_names):
        imgnames_.append(img_name)
        if centers[i][0] < 0 or centers[i][1] < 0:
            img = cv2.imread(root_path + '/' + img_name)
            img_ = img[:, :, ::-1].copy()
            box = get_box(img_)
            center, scale = location_from_box(box)
            centers_.append(center)
            scales_.append(scale)
            img_path = mask_path + img_name.split('/')[0] + '_' + img_name.split('/')[-1].split('.')[0] + '.png'
            mask_file = skech_to_mask(img, img_path)
            openpose = pose_from_mask(mask_file, parts[i], dataset)
            openposes_.append(openpose)
        else:
            centers_.append(centers[i])
            scales_.append(scales[i])
            openposes_.append(openposes[i])
        parts_.append(parts[i])
        Ss_.append(Ss[i])
        has_smpls_.append(has_smpls[i])
        shapes_.append(shapes[i])
        poses_.append(poses[i])
    np.savez(new_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           pose=poses_,
                           shape=shapes_,
                           has_smpl=has_smpls_,
                           S=Ss_,
                           openpose=openposes_)


def update_2d_kepoints(dataset):
    visualizer = Visualizer('opengl')
    file = np.load(config.DATASET_FILES[1][dataset])
    new_file = 'data/dataset_extras/mpi-inf-3dhp_train_sketch.npz'
    # 更新数据集文件center、scale、openpose
    root_path = ''
    mask_path = 'data/mask/'
    if dataset == 'lsp-orig':
        root_path = config.LSP_ROOT
    if dataset == 'mpi-inf-3dhp':
        root_path = config.MPI_INF_3DHP_ROOT
    centers = file['center']
    scales = file['scale']
    poses = file['pose']
    parts = file['part']
    shapes = file['shape']
    has_smpls = file['has_smpl']
    Ss = file['S']
    img_names = file['imgname']
    openposes = file['openpose']

    imgnames_, scales_, centers_, openposes_, parts_ = [], [], [], [], []
    Ss_, has_smpls_, shapes_, poses_ = [], [], [], []
    for i, img_name in enumerate(img_names):
        imgnames_.append(img_name)
        if centers[i][0] < 0 or centers[i][1] < 0:
            img = cv2.imread(root_path + '/' + img_name)
            img_ = img[:, :, ::-1].copy()
            box = get_box(img_)
            center, scale = location_from_box(box)
            centers_.append(center)
            scales_.append(scale)
            img_path = mask_path + img_name.split('/')[0] + '_' + img_name.split('/')[-1].split('.')[0] + '.png'
            mask_file = skech_to_mask(img, img_path)
            openpose = pose_from_mask(mask_file, parts[i], dataset)
            openposes_.append(openpose)
        else:
            centers_.append(centers[i])
            scales_.append(scales[i])
            openposes_.append(openposes[i])
        parts_.append(parts[i])
        Ss_.append(Ss[i])
        has_smpls_.append(has_smpls[i])
        shapes_.append(shapes[i])
        poses_.append(poses[i])
    np.savez(new_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             pose=poses_,
             shape=shapes_,
             has_smpl=has_smpls_,
             S=Ss_,
             openpose=openposes_)

def pose_from_mask(mask_file, dataset):  # data from file
    model_path = "/home/stu/PycharmProjects/opscoco/models"
    pose_model = general_pose_model(model_path, mode="BODY25")

    res_points = pose_model.predict(mask_file)
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]

    op_keyp25 = np.reshape(res_points, [25, 3])
    parts = op_keyp25[op_to_12]

    keypoint_25 = np.reshape(res_points, [25, 3])
    return keypoint_25, parts


def location_from_box(box):
    center = box[:2] + 0.5 * box[2:]
    bbox_size = max(box[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    return center, scale  # , bbox_XYWH


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


def get_box(img, is_train=True):
    if is_train:
        new_img = img.copy()
        new_img = new_img.tolist()
        img_point_list = []
        for i in new_img:
            y_list = []
            for j in i:
                s = sum(j)
                y_list.append(s)
            img_point_list.append(y_list[:])
        x, y, w, h = -1, -1, 0, 0
        for i in range(len(img_point_list)):
            for j in range(len(img_point_list[0])):
                if img_point_list[i][j] != 0:
                    y = i
                    break
            if y != -1:
                break
        for i in range(len(img_point_list[0])):
            for j in range(len(img_point_list)):
                if img_point_list[j][i] != 0:
                    x = i
                    break
            if x != -1:
                break
        x1 = -1
        y1 = -1
        for i in range(len(img_point_list), -1, -1):
            for j in range(len(img_point_list[0]), -1, -1):
                if img_point_list[i - 1][j - 1] != 0:
                    y1 = i
                    break
            if y1 != -1:
                break
        for i in range(len(img_point_list[0]), -1, -1):
            for j in range(len(img_point_list), -1, -1):
                if img_point_list[j - 1][i - 1] != 0:
                    x1 = i
                    break
            if x1 != -1:
                break
        w = x1 - x
        h = y1 - y
        return np.array([float(x), float(y), float(w), float(h)])
    else:
        new_img = img.copy()
        new_img = new_img.tolist()
        x, y, w, h = -1, -1, 0, 0
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                if new_img[i][j] != 0:
                    y = i
                    break
            if y != -1:
                break
        for i in range(len(new_img[0])):
            for j in range(len(new_img)):
                if new_img[j][i] != 0:
                    x = i
                    break
            if x != -1:
                break
        x1 = -1
        y1 = -1
        for i in range(len(new_img), -1, -1):
            for j in range(len(new_img[0]), -1, -1):
                if new_img[i - 1][j - 1] != 0:
                    y1 = i
                    break
            if y1 != -1:
                break
        for i in range(len(new_img[0]), -1, -1):
            for j in range(len(new_img), -1, -1):
                if new_img[j - 1][i - 1] != 0:
                    x1 = i
                    break
            if x1 != -1:
                break
        w = x1 - x
        h = y1 - y
        return np.array([float(x), float(y), float(w), float(h)])

def save_mask(mask_file, center, scale, input_res):
    mask = cv2.imread(mask_file)
    mask = crop(mask, center, scale, input_res, rot=0)
    # wh = int(max(box[2], box[3]))
    # try:
    #     # mask = mask[int(box[1]):int(box[1]+wh), int(box[0]):int(box[0]+wh), :]
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # except:
    #     print('error', mask_file)
    if mask is None:
        print(mask_file, 'is none')
        return
    try:
        cv2.imwrite(mask_file, mask)
    except:
        print(mask_file, 'save error')

if __name__ == '__main__':
    args = parser.parse_args()
    for i in range(4):
        if i == 0:
            args.dataset = 'lsp-orig'
            args.train = True
            new_data_file(args.dataset, args.train)
        # if i == 1:
        #     args.dataset = 'lsp-orig'
        #     args.train = False
        #     new_data_file(args.dataset, args.train)
        if i == 2:
            args.dataset = 'mpi-inf-3dhp'
            args.train = True
            new_data_file(args.dataset, args.train)
        # else:
        #     args.dataset = 'mpi-inf-3dhp'
        #     args.train = False
        #     new_data_file(args.dataset, args.train)
    # process_erro(args.dataset)
