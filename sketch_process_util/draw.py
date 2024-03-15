#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/3 14:22
# @Author : Chen Shan
# Function :GUI programming - a naive Sketchpad tool

import os
import sys

import numpy as np
import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import argparse

from cv2 import cv2

import utils.geometry_utils as gu

from torchvision.transforms import Normalize

import config
import constants
from models import hmr, SMPL
from models.shmr import shmr
from sketch_process import get_box, location_from_box
from utils.imutils import crop
from utils.renderer import Renderer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True,
                    help='Path to pretrained checkpoint',
                    default='logs/train_example/checkpoints/2022_08_17-14_38_30_loss_6.583424776047568.pt')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
# parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
# parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None,
                    help='Filename of output images. If not set use input filename.')

args = parser.parse_args()

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

# Load pretrained model
model = shmr(smpl_mean_params=config.SMPL_MEAN_PARAMS).to(device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model'], strict=False)

# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False).to(device)
model.eval()

# Setup renderer for visualization
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

class PenWidthDlg(QDialog):
    def __init__(self, parent=None):
        super(PenWidthDlg, self).__init__(parent)

        widthLabel = QLabel("宽度:")
        self.widthSpinBox = QSpinBox()
        widthLabel.setBuddy(self.widthSpinBox)
        self.widthSpinBox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.widthSpinBox.setRange(0, 50)

        okButton = QPushButton("ok")
        cancelButton = QPushButton("cancle")

        layout = QGridLayout()
        layout.addWidget(widthLabel, 0, 0)
        layout.addWidget(self.widthSpinBox, 0, 1)
        layout.addWidget(okButton, 1, 0)
        layout.addWidget(cancelButton, 1, 1)
        self.setLayout(layout)
        self.setWindowTitle("宽度设置")

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)


class myMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_load = False
        self.setWindowTitle("draw")
        self.pix = QPixmap()
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        # 初始化参数
        self.initData()
        # 清空画布
        self.initView()
        # 菜单栏
        self.Menu = self.menuBar().addMenu("菜单")

        # 清空
        self.ClearAction = QAction(QIcon("sketch_process/images/clear.png"), "清空", self)
        self.ClearAction.triggered.connect(self.initView)
        self.Menu.addAction(self.ClearAction)

        # 调画笔颜色
        self.changeColor = QAction(QIcon("sketch_process/images/icon.png"), "颜色", self)
        self.changeColor.triggered.connect(self.showColorDialog)
        self.Menu.addAction(self.changeColor)

        # 调画笔粗细
        self.changeWidth = QAction(QIcon("sketch_process/images/width.png"), "宽度", self)
        self.changeWidth.triggered.connect(self.showWidthDialog)
        self.Menu.addAction(self.changeWidth)

        # #右侧停靠窗口
        # logDockWidget=QDockWidget("Log",self)
        # logDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        # self.listWidget=QListWidget()
        # logDockWidget.setWidget(self.listWidget)
        # self.addDockWidget(Qt.RightDockWidgetArea,logDockWidget)

        # 各种动作
        self.fileOpenAction = QAction(QIcon("sketch_process/images/fileopen.png"), "&Open", self)
        self.fileOpenAction.setShortcut(QKeySequence.Open)
        self.fileOpenAction.setToolTip("Open an image.")
        self.fileOpenAction.setStatusTip("Open an image.")
        self.fileOpenAction.triggered.connect(self.fileOpen)

        self.fileSaveAction = QAction(QIcon("sketch_process/images/filesave.png"), "&Save", self)
        self.fileSaveAction.setShortcut(QKeySequence.Save)
        self.fileSaveAction.setToolTip("Save an image.")
        self.fileSaveAction.setStatusTip("Save an image.")
        self.fileSaveAction.triggered.connect(self.fileSaveAs)

        # 工具栏
        fileToolbar = self.addToolBar("文件")
        fileToolbar.addAction(self.fileOpenAction)
        fileToolbar.addAction(self.fileSaveAction)

        editToolbar = self.addToolBar("清空")
        editToolbar.addAction(self.ClearAction)

        colorToolbar = self.addToolBar("颜色")
        colorToolbar.addAction(self.changeColor)

        widthToolbar = self.addToolBar("宽度")
        widthToolbar.addAction(self.changeWidth)

        # 状态栏
        self.sizeLabel = QLabel()
        self.sizeLabel.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.sizeLabel)
        status.showMessage("Ready", 5000)

    def initData(self):
        self.size = QSize(1000, 1040)
        self.pixmap = QPixmap(self.size)

        self.dirty = False
        self.filename = None
        self.recentFiles = []

        # 新建画笔
        self.width = 5
        self.color = QColor(255, 255, 255)
        self.pen = QPen()  # 实例化画笔对象
        self.pen.setColor(self.color)  # 设置画笔颜色
        self.pen = QPen(Qt.SolidLine)  # 实例化画笔对象.参数：画笔样式
        self.pen.setWidth(self.width)  # 设置画笔粗细

        # 新建绘图工具
        self.painter = QPainter(self.pixmap)
        self.painter.setPen(self.pen)

        # 鼠标位置
        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.image = QImage()

    def initView(self):
        # 设置界面的尺寸为__size
        self.Clear()
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(self.pixmap)
        self.setCentralWidget(self.imageLabel)

    def Clear(self):
        # 清空画板
        self.is_load = False
        self.pixmap.fill(Qt.white)
        self.update()
        self.dirty = False
        # self.initView()

    def mousePressEvent(self, event):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置

        pointX = event.globalX()
        pointY = event.globalY()
        self.__currentPos = QPoint(pointX, pointY)
        self.dirty = True
        self.__currentPos = event.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, event):

        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = event.pos()
        # pointX = event.globalX()
        # pointY = event.globalY()
        # self.__currentPos = QPoint(pointX,pointY)

        # 画线
        # 用begin和end抱起来，表示针对这个对象，就可以在pixmap有图的情况下继续画画
        self.painter.begin(self.pixmap)

        self.painter.setPen(self.pen)
        self.painter.drawLine(self.__lastPos, self.__currentPos)

        self.__lastPos = self.__currentPos
        self.painter.end()
        self.update()  # 更新显示
        self.imageLabel.setPixmap(self.pixmap)

    # 调画笔颜色
    def showColorDialog(self):
        col = QColorDialog.getColor()
        self.pen.setColor(col)
        self.painter.setPen(self.pen)

    def updateWidth(self):
        self.pen.setWidth(self.width)
        self.painter.setPen(self.pen)

    def showWidthDialog(self):
        dialog = PenWidthDlg(self)
        dialog.widthSpinBox.setValue(self.width)
        if dialog.exec_():
            self.width = dialog.widthSpinBox.value()
            self.updateWidth()

    ###########################################################
    def okToContinue(self):  # 警告当前图像未保存
        if self.dirty:
            reply = QMessageBox.question(self,
                                         "Image Changer - Unsaved Changes",
                                         "图片已被更改，请问要保存吗?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                return self.fileSaveAs()
        return True

    def fileOpen(self):
        self.is_load = True
        if not self.okToContinue():
            return
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")
        formats = (["*.{}".format(format.data().decode("ascii").lower())
                    for format in QImageReader.supportedImageFormats()])
        fname = QFileDialog.getOpenFileName(self,
                                            "Image Changer - Choose Image", dir,
                                            "Image files ({})".format(" ".join(formats)))
        if fname:
            print(fname[0])
            self.loadFile(fname[0])
            self.updateFileMenu()

    def loadFile(self, fname=None):
        self.is_load = True
        if fname is None:
            action = self.sender()
            if isinstance(action, QAction):
                fname = action.data()
                if not self.okToContinue():
                    return
            else:
                return
        if fname:
            self.filename = None
            image = QImage(fname)
            if image.isNull():
                message = "Failed to read {}".format(fname)
            else:
                self.addRecentFile(fname)
                self.image = QImage()
                # self.editUnMirrorAction.setChecked(True)
                self.image = image
                self.filename = fname
                self.showImage()
                self.dirty = False
                self.sizeLabel.setText("{} x {}".format(
                    image.width(), image.height()))
                message = "Loaded {}".format(os.path.basename(fname))
            self.updateStatus(message)

    def updateStatus(self, message):
        self.statusBar().showMessage(message, 5000)
        # self.listWidget.addItem(message)
        if self.filename:
            self.setWindowTitle("Image Changer - {}[*]".format(
                os.path.basename(self.filename)))
        elif not self.image.isNull():
            self.setWindowTitle("Image Changer - Unnamed[*]")
        else:
            self.setWindowTitle("Image Changer[*]")
        self.setWindowModified(self.dirty)

    def updateFileMenu(self):
        self.Menu.clear()
        self.Menu.addAction(self.fileOpenAction)
        self.Menu.addAction(self.fileSaveAction)
        current = self.filename
        recentFiles = []
        print(self.recentFiles)
        for fname in self.recentFiles:
            if fname != current and QFile.exists(fname):
                recentFiles.append(fname)
        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QAction(QIcon("images/icon.png"),
                                 "&{} {}".format(i + 1, QFileInfo(
                                     fname).fileName()), self)
                action.setData(fname)
                action.triggered.connect(lambda: self.loadFile(fname))
                self.fileMenu.addAction(action)

    def addRecentFile(self, fname):
        if fname is None:
            return
        if fname not in self.recentFiles:
            if len(self.recentFiles) < 10:
                self.recentFiles = [fname] + self.recentFiles
            else:
                self.recentFiles = [fname] + self.recentFiles[:8]
            print(len(self.recentFiles))

    def fileSaveAs(self):
        # savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        savePath = ['sketch_process/result.png']
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

        def QImageToCvMat(incomingImage):
            '''  Converts a QImage into an opencv MAT format  '''

            incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

            width = incomingImage.width()
            height = incomingImage.height()

            ptr = incomingImage.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            return arr

        print("save...")
        self.pixmap.save(savePath[0])
        self.updateStatus("Saved as {}".format(savePath))

        # image = QImageToCvMat(self.pixmap.toImage())
        image = savePath[0]
        img_obj(image, self.is_load)
        # 开始处理图像

    def showImage(self, percent=None):
        if self.image.isNull():
            return
        self.pixmap = QPixmap.fromImage(self.image)
        self.imageLabel.setPixmap(self.pixmap)

def img_obj(img_file, is_load):
    # Preprocess input image and generate predictions
    print('process img start')
    img, org_img, norm_img, center, scale= process_image(img_file, input_res=constants.IMG_RES, is_load=is_load)
    with torch.no_grad():
        print('model start')
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        print('model end')
        pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).to(device)
        pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
        body_pose = pred_aa[:, 3:]
        global_orient = pred_aa[:, :3]

        # 得到mesh
        print('using model start')
        pred_output = smpl(betas=pred_betas, body_pose=body_pose,
                           global_orient=global_orient, pose2rot=False)
        # camera = pred_camera.cpu().numpy().ravel()
        #
        # new_img, boxScale_o2n, bboxTopLeft = imutils.crop_bboxInfo(org_img,
        #                                                        center, scale, (224, 224))
        # pred_vertices_bbox = convert_smpl_to_bbox(pred_output.vertices[0].cpu().detach().numpy(), camera[0], camera[1:])
        # pred_vertices_img = convert_bbox_to_oriIm(
        #     pred_vertices_bbox, boxScale_o2n, bboxTopLeft, org_img.shape[1], org_img.shape[0])
        #
        # # # 导出mesh_2d
        # out_mesh = trimesh.Trimesh(pred_vertices_img, smpl.faces.astype(np.int32), process=True)
        # rot_mesh = trimesh.transformations.rotation_matrix(
        #     np.radians(180), [1, 0, 0])
        # out_mesh.apply_transform(rot_mesh)
        # out_mesh.show()
        # out_mesh.export("result.obj")
        pred_vertices = pred_output.vertices
        print('using model end')

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                      2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                     dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = img.permute(1, 2, 0).cpu().numpy()

    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

    # mesh = trimesh.load("result.obj")
    # mesh.show()

    outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
    shape_img = 255 * img_shape[:, :, ::-1]

    shape_img = cv2.resize(shape_img, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
    # Save reconstructions
    print('save img')
    cv2.imwrite('shape.png', 255 * img_shape[:, :, ::-1])
    cv2.imshow('shape.png', shape_img)
    cv2.waitKey(0)
    cv2.imwrite('shape_side.png', 255 * img_shape_side[:, :, ::-1])
    print('end')

def process_image(img_file, input_res=224, is_load=False):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    #     If there are bounding box annotations, use them to crop the image.
    #     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    #     """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    org_img = cv2.imread(img_file)
    # 颜色反转
    if not is_load:
        org_img = 255 - org_img
    img = org_img[:, :, ::-1].copy()  # PyTorch does not support negative stride at the moment
    box = get_box(img)
    center, scale = location_from_box(box)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, org_img, norm_img, center, scale

app = QApplication(sys.argv)
form = myMainWindow()
form.setMinimumSize(1000, 1000)
form.show()
app.exec_()

