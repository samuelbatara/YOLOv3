from tkinter.tix import IMAGE
import torch

DATASET = "VOC"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20 
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
S = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_FILENAME = "model.pth.tar"
IMG_DIR = "/images"
LABEL_DIR = "/labels"
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]    
] # Rescale menjadi [0..1] (dibagi dengan imaga_size=416)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
