"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr
from models.hmr.hmr import HMRNetBase
from models.hpr import hpr
from models.smpl import SMPL
from models import shpr
from datasets import BaseDataset
from models.vhpr import vhpr

from utils.imutils import uncrop, crop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
import utils.geometry_utils as gu

from sketch_process_util.sketch_process import get_box, location_from_box

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='logs/comparison/hmr/checkpoints/2022_11_22-23_39_39_loss_0.3023701024057326.pt',
                    help='Path to network checkpoint')
parser.add_argument('--dataset', default='sketch-mpi-inf-3dhp', choices=['sketch-mpi-inf-3dhp', 'sketch-lsp'],
                    help='Choose evaluation dataset')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--log_freq', default=10000, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=64, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=3, type=int, help='Number of processes for data loading')
parser.add_argument('--img_res', default=224, type=int, help='Number of processes for data loading')
parser.add_argument("--model_name", type=str, default='shape',
                    choices=['shape', 'pose', 'hmr', 'graph_cnn', 'spin', '..'], help="model name")
parser.add_argument('--pose_checkpoint_file', default='logs/train_example/history/shpr/shpr-finnaly-pose-best.pt',
                    help='Load a pretrained Graph CNN when starting training')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')


class Eval():
    def __init__(self, options):
        self.options = options
        self.model = HMRNetBase(options.batch_size)

    def eval_look(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

        for i in range(2):
            if i == 0:
                dataset_name = 'sketch-lsp'
            else:
                dataset_name = 'sketch-mpi-inf-3dhp'
            # Setup evaluation dataset
            dataset = BaseDataset(None, dataset_name, is_train=False)
            # Run evaluation
            run_evaluation(self.model, dataset_name, dataset, result_file=None,
                           batch_size=self.options.batch_size,
                           shuffle=False,
                           log_freq=10000)


def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=1, img_res=224,
                   num_workers=0, shuffle=False, log_freq=50,
                   show=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(device)

    renderer = PartRenderer()

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle = False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_cnn = np.zeros(len(dataset))
    recon_err_cnn = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_cnn = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    accuracy_cnn = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2, 1))
    fp = np.zeros((2, 1))
    fn = np.zeros((2, 1))
    # Pixel count accumulators
    pixel_count = 0
    pixel_count_cnn = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    eval_shape = False
    eval_gnn = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'sketch-mpi-inf-3dhp' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
        # eval_masks = True
    elif dataset_name == 'sketch-lsp' or dataset_name == 'lsp':
        eval_masks = True
    elif dataset_name == 'sketch-up-3d':
        eval_shape = True

    joint_mapper_h36m = constants.H36M_TO_J17
    joint_mapper_gt = constants.J24_TO_J17
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['beta'].to(device)
        gt_output = smpl_neutral(gt_pose, gt_betas)
        gt_vertices = gt_output.vertices
        pred_vertices = torch.zeros_like(gt_vertices)
        pred_vertices_cnn = torch.zeros_like(gt_vertices)
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_vertices_cnn = smpl_neutral(pred_rotmat, pred_betas).vertices
            mask_cnn, _ = renderer(pred_vertices_cnn, pred_camera, False)
            if show:
                cv2.imshow('w', images.cpu().numpy()[0][0])
                cv2.imshow('r', mask_cnn.cpu().numpy()[0]*255)

        if save_results:
            rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :] = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :] = pred_camera.cpu().numpy()

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints

            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]

            # Get 14 predicted joints from the mesh
            if eval_gnn:
                pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_keypoints_3d_cnn = torch.matmul(J_regressor_batch, pred_vertices_cnn)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            if eval_gnn:
                pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
                pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_pelvis_cnn = pred_keypoints_3d_cnn[:, [0], :].clone()
            pred_keypoints_3d_cnn = pred_keypoints_3d_cnn[:, joint_mapper_h36m, :]
            pred_keypoints_3d_cnn = pred_keypoints_3d_cnn - pred_pelvis_cnn

            # Absolute error (MPJPE)
            if eval_gnn:
                error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            error_cnn = torch.sqrt(((pred_keypoints_3d_cnn - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            mpjpe_cnn[step * batch_size:step * batch_size + curr_batch_size] = error_cnn
            # Reconstuction_error
            if eval_gnn:
                r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                               reduction=None)
                recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error
            r_error_cnn = reconstruction_error(pred_keypoints_3d_cnn.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                               reduction=None)
            recon_err_cnn[step * batch_size:step * batch_size + curr_batch_size] = r_error_cnn

        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            if eval_gnn:
                mask, _ = renderer(pred_vertices, pred_camera, False)
            mask_cnn, _ = renderer(pred_vertices_cnn, pred_camera, False)

        # Mask evaluation (for LSP)
        if eval_masks:
            for i in range(curr_batch_size):
                # Load gt mask
                gt_mask = cv2.imread(batch['maskname'][i], 0)
                box = get_box(gt_mask, False)
                center, scale = location_from_box(box)
                gt_mask = crop(gt_mask, center, scale, (224, 224))
                if show:
                    cv2.imshow('gt_mask', gt_mask)
                    cv2.waitKey(0)
                gt_mask = gt_mask > 0
                if eval_gnn:
                    pred_mask = mask[i].cpu().numpy().astype(int)
                    box = get_box(pred_mask, False)
                    center, scale = location_from_box(box)
                    pred_mask = crop(pred_mask, center, scale, (224, 224))
                    # Evaluation consistent with the original UP-3D code
                    accuracy += (gt_mask == pred_mask).sum()
                    pixel_count += np.prod(np.array(gt_mask.shape))
                    for c in range(2):
                        cgt = gt_mask == c
                        cpred = pred_mask == c
                        tp[c] += (cgt & cpred).sum()
                        fp[c] += (~cgt & cpred).sum()
                        fn[c] += (cgt & ~cpred).sum()
                    f1 = 2 * tp / (2 * tp + fp + fn)
                pred_mask_cnn = mask_cnn[i].cpu().numpy().astype(int)
                box = get_box(pred_mask_cnn, False)
                center, scale = location_from_box(box)
                pred_mask_cnn = crop(pred_mask_cnn, center, scale, (224, 224))
                pred_mask_cnn = pred_mask_cnn > 0
                accuracy_cnn += (gt_mask == pred_mask_cnn).sum()
                pixel_count_cnn += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask_cnn == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] += (~cgt & cpred).sum()
                    fn[c] += (cgt & ~cpred).sum()
                f1_cnn = 2 * tp / (2 * tp + fp + fn)

        # Shape evaluation (Mean per-vertex error)
        if eval_shape:
            se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            se_cnn = torch.sqrt(((pred_vertices_cnn - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            shape_err[step * batch_size:step * batch_size + curr_batch_size] = se
            shape_err_cnn[step * batch_size:step * batch_size + curr_batch_size] = se_cnn

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                if eval_gnn:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('MPJPE CNN: ' + str(1000 * mpjpe_cnn[:step * batch_size].mean()))
                print('Reconstruction Error CNN: ' + str(1000 * recon_err_cnn[:step * batch_size].mean()))
                print()
            if eval_masks:
                if eval_gnn:
                    print('Accuracy: ', accuracy / pixel_count)
                    print('F1: ', f1.mean())
                print('Accuracy CNN: ', accuracy_cnn / pixel_count_cnn)
                print('F1 CNN: ', f1_cnn.mean())
                print()
            if eval_shape:
                if eval_gnn:
                    print('Shape Error: ' + str(1000 * shape_err[:step * batch_size].mean()))
                print('Shape Error (CNN): ' + str(1000 * shape_err_cnn[:step * batch_size].mean()))
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    result = {}
    if eval_pose:
        if eval_gnn:
            print('MPJPE: ' + str(1000 * mpjpe.mean()))
            print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('MPJPE CNN: ' + str(1000 * mpjpe_cnn.mean()))
        print('Reconstruction Error CNN: ' + str(1000 * recon_err_cnn.mean()))
        print()
    if eval_masks:
        if eval_gnn:
            print('Accuracy: ', accuracy / pixel_count)
            print('F1: ', f1.mean())
        print('Accuracy CNN: ', accuracy_cnn / pixel_count_cnn)
        print('F1 CNN: ', f1_cnn.mean())
        print()
    if eval_shape:
        if eval_gnn:
            print('Shape Error: ' + str(1000 * shape_err.mean()))
        print('Shape Error (CNN): ' + str(1000 * shape_err_cnn.mean()))
        print()


if __name__ == '__main__':
    args = parser.parse_args()

    model = HMRNetBase(args.batch_size)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()

    for i in range(4):
        if i == 0:
            args.dataset = 'sketch-lsp'
        elif i == 1:
            args.dataset = 'sketch-mpi-inf-3dhp'
        elif i == 2:
            args.dataset = 'lsp'
        elif i == 3:
            args.dataset = 'mpi-inf-3dhp'
        # Setup evaluation dataset
        dataset = BaseDataset(None, args.dataset, is_train=False)
        # Run evaluation
        run_evaluation(model, args.dataset, dataset, args.result_file,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle,
                       log_freq=args.log_freq)
