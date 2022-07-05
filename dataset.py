import torch 
import torch.nn as nn
import cv2
import os
from PIL import Image
import pandas as pd

from utils import iou_width_height

class YoloDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir 
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors_per_scale = self.anchors.shape[0] // len(S)
        self.S = S
        self.C = C
        self.B = 3
        self.transform = transform
        self.ignore_iou_threshold = 0.5
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        # Memuat bounding box
        label_path = os.path.join(self.label_dir, self.csv_file.iloc[index, 1])
        bboxes = self.get_bboxes(label_path)

        # Memuat gambar
        img_path = os.path.join(self.img_dir, self.csv_file.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if(self.transform):
            img = self.transform(img)

        # targets.shape = (jumlah S, jumlah anchor per S, jumlah bounding box, S, S, 6)
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]
        for box in bboxes:
            class_label, x, y, width, height = box
            iou_anchors = iou_width_height(
                torch.tensor([width, height]),
                self.anchors,
            )
            anchor_indexes = iou_anchors.argsort(descending=True, dim=0)
            has_anchor = [False, False, False] 

            for anchor_idx in anchor_indexes:
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if(not anchor_taken and not has_anchor[scale_idx]):
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell = S * width
                    height_cell = S * height
                    box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinate
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif(not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold):
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        
        return img, targets

    def get_bboxes(self, label_path):
        bboxes = []
        
        with open(label_path, "r") as f:
            for line in f.readlines():
                 bboxes.append(
                     list(map(float, line.split(' ')))
                 )
        
        return bboxes