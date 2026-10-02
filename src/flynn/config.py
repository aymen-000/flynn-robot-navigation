
from __future__ import annotations

import os
from pathlib import Path

import torch

REPO_ROOT = Path(os.environ.get("FLYNN_ROOT", Path(__file__).resolve().parents[2])).resolve()

ASSETS_DIR = REPO_ROOT / "assets"
MUJOCO_ASSETS_DIR = ASSETS_DIR / "mujoco"          # MJCF XML files + textures/
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"          # *.pt model weights
LOSS_DIR = OUTPUT_DIR / "loss"                       # per-run training logs / loss curves
RESULTS_DIR = OUTPUT_DIR / "results"                 # comparison tables and plots
VIDEO_DIR = OUTPUT_DIR / "videos"                    # rollout videos


TRAIN_RNN_WEIGHTS = True
TRAIN_RNN_BIAS = True
TRAIN_READOUT_HEAD = True
TRAIN_VALUE_HEAD = True
TRAIN_INPUT_SCALE = True
TRAIN_POLICY_HEAD = True
TRAIN_WIND_MLP = True
TRAIN_RETINA = True


LR = 3e-4


CONNECTOME_NAME = os.environ.get("FLYNN_CONNECTOME", "drosophila_adult")

_EDGE_FILE_BY_CONNECTOME = {
    "drosophila_adult": "connections_princeton.csv",
    "ws_small_world": "connections_ws_small_world.csv",
}
if CONNECTOME_NAME not in _EDGE_FILE_BY_CONNECTOME:
    raise ValueError(f"Unknown connectome {CONNECTOME_NAME!r}; expected one of {sorted(_EDGE_FILE_BY_CONNECTOME)}")

CONNECTOME_DIR = DATA_DIR / "connectomes" / CONNECTOME_NAME
EDGE_PATH = str(CONNECTOME_DIR / _EDGE_FILE_BY_CONNECTOME[CONNECTOME_NAME])

PHOTORECEPTOR_LEFT_CSV = str(CONNECTOME_DIR / "visual_column_L1_L2_L3_rear_view_left.csv")
PHOTORECEPTOR_RIGHT_CSV = str(CONNECTOME_DIR / "visual_column_L1_L2_L3_rear_view_right.csv")
TACTILE_LEFT_CSV = str(CONNECTOME_DIR / "head_bristles_left.csv")
TACTILE_RIGHT_CSV = str(CONNECTOME_DIR / "head_bristles_right.csv")
DESCENDING_NEURONS_CSV = str(CONNECTOME_DIR / "descending_neurons.csv")
CELL_TYPES_CSV = str(CONNECTOME_DIR / "consolidated_cell_types.csv")
WIND_SENSING_CSV = str(CONNECTOME_DIR / "JO-C_and_JO-E.csv")

# -----------------------------
# Environment & Model Settings
# -----------------------------
ENV_WIDTH = 128
ENV_HEIGHT = 128
MAX_EPISODE_STEPS = 600
N_OBSTACLES = 20
ARENA_HALF_EXTENT = 7.0
RENDER_MODE = None
END_ON_COLLISION = False

LEAK_ALPHA = 0.2
ACTIVATION = "tanh"
TARGET_RHO = 0.9
BATCH_CHUNK = 8
ROW_TILE_SIZE = 69320
INPUT_SCALE_INIT = 1.0
DTYPE = torch.float32
USE_GRADIENT_CHECKPOINT = False
