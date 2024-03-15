"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

SKETCH_LSP_ORIGINAL_ROOT = 'datasets/data/sketch-lsp'
SKETCH_MPI_INF_3DHP_ROOT = 'datasets/data/sketch-mpi-inf-3dhp'
SKETCH_UP_3D_ROOT = 'datasets/data/up-3d'
HU36M_ROOT = 'datasets/data/h36m'

# Path to test/train npz files
DATASET_FILES = [{'sketch-lsp': join(DATASET_NPZ_PATH, 'lsp-orig-valid-sketch.npz'),
                  'sketch-mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi-inf-3dhp-valid-sketch.npz'),
                  'up-3d': join(DATASET_NPZ_PATH, 'up-3d-valid-canny.npz'),
                  'lsp': join(DATASET_NPZ_PATH, 'lsp-orig-valid-canny.npz'),
                  'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi-inf-3dhp-valid-canny.npz')
                  },
                 {'sketch-mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi-inf-3dhp-train-sketch.npz'),
                  'up-3d': join(DATASET_NPZ_PATH, 'up-3d-train-canny.npz'),
                  'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi-inf-3dhp-train-canny.npz')
                  }
                 ]

DATASET_FOLDERS = {'sketch-lsp': SKETCH_LSP_ORIGINAL_ROOT,
                   'sketch-mpi-inf-3dhp': SKETCH_MPI_INF_3DHP_ROOT,
                   'lsp': SKETCH_LSP_ORIGINAL_ROOT,
                   'mpi-inf-3dhp': SKETCH_MPI_INF_3DHP_ROOT,
                   'up-3d': SKETCH_UP_3D_ROOT,
                   }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
SMPL_FILE = 'data/smpl/SMPL_NEUTRAL.pkl'

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
