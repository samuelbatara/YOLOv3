import torch

ANCHORS = [[(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)], # scale: 13
           [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # scale: 26
           [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # scale: 52
           ]
S  = [13, 26, 52]
IMG_SIZE = 416
IMG_DIR = "../input/vehicle-detection/Vehicle Detection/images"
LABEL_DIR = "../input/vehicle-detection/Vehicle Detection/labels"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
TRAIN_SIZE = 500
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"