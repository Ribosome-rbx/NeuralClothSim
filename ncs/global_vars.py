import os

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCANDATASET_DIR = os.path.join(ROOT_DIR, "scan_data")
BODY_DIR = os.path.join(ROOT_DIR, "body_models")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR = '/cluster/scratch/borong/'
TXT_DIR = os.path.join(ROOT_DIR, "ncs", "dataset", "txt")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SMPL_DIR = os.path.join(ROOT_DIR, "smpl")
TMP_DIR = os.path.join(ROOT_DIR, "tmp")

# Skeletons
NUM_JOINTS = {"smpl": 24, "mixamo": 65, "smplx": 55}

# Physick
GRAVITY = 9.81
