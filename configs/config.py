import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 7
IMAGE_SIZE = 224

BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

BASE_DIR = "cattle diseases.v2i.multiclass"

TRAIN_DIR = "./cattle diseases.v2i.multiclass/train"
VAL_DIR = "./cattle diseases.v2i.multiclass/valid"
TEST_DIR = "./cattle diseases.v2i.multiclass/test"

TRAIN_EXCEL = "balanced_splits/train.csv"
VAL_EXCEL   = "balanced_splits/valid.csv"
TEST_EXCEL  = "balanced_splits/test.csv"

# ========================
# RESULTS PATHS
# ========================
MODEL_SAVE_PATH = "results/models/best_model.pth"
FUSION_MODEL_PATH = "results/models/fusion_model.pth"
THRESHOLD_PATH = "results/thresholds/thresholds.npy"
FUSION_THRESHOLD_PATH = "results/thresholds/fusion_thresholds.npy"