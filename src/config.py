"""Central configuration. Every value can be overridden from the command line."""
import os

# ---- Paths (APTOS 2019 layout: train.csv + train_images/*.png) ----
DATA_DIR = os.environ.get("DR_DATA_DIR", "data/aptos2019")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
RAW_IMG_DIR = os.path.join(DATA_DIR, "train_images")
PREPROC_DIR = os.path.join(DATA_DIR, "preprocessed")
OUTPUT_DIR = "outputs"

# ---- Task ----
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# ---- Model ----
IMG_SIZE = 384
DROPOUT = 0.3

# ---- Training ----
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PATIENCE = 8          # early stopping on validation QWK
VAL_SPLIT = 0.15   # plus 0.15 test, 0.70 train
SEED = 42
NUM_WORKERS = min(4, os.cpu_count() or 2)
