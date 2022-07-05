import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.lambda_class = 1

    def forward(self, predictions, targets, anchors):
        # 1 -> ada objek
        # 0 -> tidak ada objek
        # -1 -> abaikan
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        # LOSS UNTUK TIDAK ADA OBJEK
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], targets[..., 0:1][noobj]
        )

        # LOSS UNTUK ADA OBJEK
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors],
            dim=-1
        )
        ious = intersection_over_union(box_preds[obj], targets[..., 1:5][obj])
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]),
            ious * targets[..., 0:1][obj]
        )

        # LOSS UNTUK KOORDINAT KOTAK
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5]/anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        # LOSS UNTUK KLASIFIKASI KELAS
        class_loss = self.entropy(
            predictions[..., 5:][obj], targets[..., 5][obj].long()
        )

        # Total loss
        loss = (
            self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss 
            + self.lambda_box * box_loss
            + self.lambda_class * class_loss
        ) 

        return loss