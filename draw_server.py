# coding:utf-8
import base64

from flask import Flask, Response
from flask import jsonify
from flask import request
from flask_cors import *

import cv2
import torch
import numpy as np
import trimesh
from torchvision.transforms import Normalize

import constants
from models.smpl import SMPL
from models.hpr_attention import hpr_attention_net
from utils.imutils import crop

device = torch.device('cpu')
smpl = SMPL('data/smpl',
            batch_size=1,
            create_transl=False).to(device)
hpr_net = hpr_attention_net('data/smpl_mean_params.npz').to(device)
checkpoint = torch.load('logs/ablation/attention and dataset/has_attention_best/syn and free/checkpoints/sketchBodyNet-best-145-89.pt', map_location=device)
hpr_net.load_state_dict(checkpoint['model'], strict=False)
hpr_net.eval()
print('加载成功')


def get_box(img):
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


def location_from_box(box):
    center = box[:2] + 0.5 * box[2:]
    bbox_size = max(box[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    return center, scale  # , bbox_XYWH


def process_image(img_file, input_res=224, is_load=False):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    #     If there are bounding box annotations, use them to crop the image.
    #     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    #     """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    org_img = img_file
    #
    # cv2.imshow('org_img', org_img)

    # 颜色反转
    if not is_load:
        if org_img[0][0][0] == 255:
            org_img = 255 - org_img
    else:
        if org_img[0][0][0] == 255:
            org_img = 255 - org_img
    img = org_img[:, :, ::-1].copy()  # PyTorch does not support negative stride at the moment
    box = get_box(img)
    center, scale = location_from_box(box)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, org_img, norm_img, center, scale


# 创建对象
app = Flask(__name__)
# 设置跨域
CORS(app, supports_credentials=True)


# 构造post请求
@app.route("/construct", methods=["POST"])
def construct():
    file = request.form.get('image')
    filename = request.form.get('filename')
    imgname = filename + '.png'
    data_url = str.split(file, ',')[1]
    img_data = base64.urlsafe_b64decode(data_url + '=' * (4 - len(data_url) % 4))
    img_data = np.frombuffer(img_data, np.uint8)
    img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    root = 'OBJViewer-master/'
    # request.form.get：获取post请求的参数，
    # imgname = 'canvas.png'
    with torch.no_grad():
        _, _, norm_img, _, _ = process_image(img_arr, input_res=constants.IMG_RES)
        pose, shape, _ = hpr_net(norm_img.to(device))
        pred_output = smpl(pose, shape)
        # # 导出mesh_2d
        out_mesh = trimesh.Trimesh(pred_output.vertices[0].cpu().numpy(), smpl.faces.astype(np.int32), process=True)
        rot_mesh = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot_mesh)
        out_mesh.export(root + 'obj/' + filename + ".obj")
        cv2.imwrite(root + 'img/' + imgname, img_arr)
    return jsonify({"code": 200, "message": "访问成功"})
    # except:
    #     return jsonify({"code": 404, "message": "访问错误，请重试"})


@app.route("/download", methods=["GET"])
def download():
    """读取本地文件接口"""
    filename = request.args.get('filename') + '.obj'
    path = 'OBJViewer-master/obj/' + filename
    with open(path, "rb") as f:
        stream = f.read()
    response = Response(stream, content_type='application/octet-stream')
    response.headers['Content-disposition'] = 'attachment; filename=result.obj'

    return response


if __name__ == '__main__':
    # 默认方式启动
    # app.run()
    # 解决jsonify中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 以调试模式启动,host=0.0.0.0 ,则可以使用127.0.0.1、localhost以及本机ip来访问
    app.run(host="10.33.30.37", port=8899, debug=False)
