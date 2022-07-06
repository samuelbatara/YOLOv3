import config
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import numpy as np 
import torch
from collections import Counter 
from tqdm import tqdm
from merge_sort import merge_sort, left_bound
import config

def iou_width_height(boxes1, boxes2):
    """
    Menghitung intersection over union dari lebar dan panjang

    Parameters:
        boxes1 (tensor): lebar dan panjang dari kotak pertama
        boxes2 (tensor): lebar dan panjang dari kotak kedua

    Returns:
        tensor: intersection over union dari kotak pertama dan kedua
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0]  * boxes2[..., 1] - intersection

    return intersection / union

def intersection_over_union(boxes1, boxes2):
    """
    Menghitung intersection over union

    Parameters:
        boxes1 (tensor): koordinat kotak pertama (x, y, w, h)
        boxes2 (tensor): koordinat kotak kedua (x, y, w, h)

    Returns:
        tensor: intersection over union dari kotak pertama dan kedua
    """

    box1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3]/2
    box1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4]/2
    box1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3]/2
    box1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4]/2
    box2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3]/2
    box2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4]/2
    box2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3]/2
    box2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4]/2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1 + box2 - intersection + 1e-6

    return intersection/union

def non_max_suppression(bboxes, iou_threshold, threshold, num_classes):
    """
    Mengurangi jumlah bounding box dengan teknik non max suppression

    Parameters:
        bboxes (list): kumpulan bounding box (class_pred, score, x, y, w, h)
        iou_threshold (float): bounding box saling berdekatan
        threshold (float): untuk mengabaikan bounding box

    Returns:
        list: bounding box setelah dikurangi
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x : x[1], reverse=True)
    nms = [[] for _ in range(num_classes)]

    for boxA in bboxes:
        class_idx = int(boxA[0])
        assert class_idx < num_classes, "[non_max_suppression]: Jumlah kelas tidak sesuai!"
        ok = True
        
        for boxB in nms[class_idx]:
            iou = intersection_over_union(
                torch.tensor(boxA[2:]),
                torch.tensor(boxB[2:])
            )
            if(iou > iou_threshold):
                ok = False
                break
        if(ok):
            nms[class_idx].append(boxA)
    
    results = []
    for i in range(num_classes):
        for result in nms[i]:
            results.append(result)
    
    return results

def mean_average_precision(pred_boxes, true_boxes, iou_threshold, num_classes):
    """
    Menghitung mean average precision

    Paramaters:
        pred_boxes (tensor): kotak prediksi (train_idx, class_pred, score, x, y, w, h)
        true_boxes (tensor): kotak target (train_idx, class_pred, score, x, y, w, h)
        iou_threshold (float): threshold dimana kotak prediksi benar
        num_classes (int): jumlah kelas

    Returns:
        float: nilai maP berdasarkan nilai iou_threshold
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        predictions = []
        ground_truths = []

        for prediction in pred_boxes:
            if(prediction[1] == c):
                predictions.append(prediction)

        for gt in true_boxes:
            if(gt[1] == c):
                ground_truths.append(gt)
        
        amount_boxes = Counter([gt[0] for gt in ground_truths])
        for key, value in amount_boxes.items():
            amount_boxes[key] = torch.tensor(value)

        predictions.sort(key=lambda x : x[2], reverse=True)
        TP = torch.zeros(len(predictions))
        FP = torch.zeros(len(predictions))
        total_true_boxes = len(ground_truths)

        if(total_true_boxes==0):
            continue

        for prediction_idx, prediction in enumerate(predictions):
            ground_truth_img = [
                gt for gt in ground_truths if gt[0] == prediction[0]
            ]
            best_iou = 0
            best_idx = -1
            for gt_idx, gt in enumerate(ground_truths):
                iou = intersection_over_union(
                    torch.tensor(prediction[3:]),
                    torch.tensor(gt[3:])
                )

                if(iou > best_iou):
                    best_iou = iou
                    best_idx = gt_idx

            if(best_iou > iou_threshold):
                if(amount_boxes[prediction[0]][best_idx]==0):
                    amount_boxes[prediction[0]][best_idx] = 1
                    TP[prediction_idx] = 1
                else:
                    FP[prediction_idx] = 1
            else:
                FP[prediction_idx] = 1
            
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat([torch.tensor(1), precisions])
        recalls = torch.cat([torch.tensor(0), precisions])
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def plot_image(image: any, bboxes: list, img_size=10):
    """
    Menampilkan bounding box dan label pada gambar

    Parameters:
        image (list): gambar
        bboxes (list): bounding box untuk setiap objek pada gambar
    
    Returns:
        void
    """

    img = np.array(image)
    height, width, _ = img.shape 

    fig = plt.figure(figsize=(img_size, img_size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    
    for bbox in bboxes:
        x, y, w, h = bbox[2:] if len(bbox)==6 else bbox[1:]
        
        # x dan y berada di pojok kiri atas
        x -= w/2
        y -= h/2

        # mengubah posisi 
        x *= width 
        y *= height
        w *= width 
        h *= height
 
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w, h,
                edgecolor="red",
                facecolor="none",
                linewidth=1,
            )
        )

        plt.text(
            x, y-4,
            "Vehicle",
            fontdict=dict(color="white"),
            bbox=dict(facecolor="red", edgecolor="red", pad=0) 
        )

    plt.show()

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    num_classes,
    device="cuda",
):
    """
    Mendapatkan kotak prediksi yang telah dikurangi dengan non max suppression
    dan kotak target.
    Kotak prediksi dan target dapat digunakan untuk menghitung mean average precision

    Parameters:
        loader (torch.utils.data.DataLoader): data loader
        model (nn.Module): model YOLOv3
        iou_threshold (float): iou threshold pada non max suppression
        anchors (list): anchors
        threshold (float): threshold pada non max suppression
        num_classes (int): jumlah kelas
        device: "cuda" atau "cpu"

    Returns:
        tensor: bounding box yang dapat dipakai untuk plot image
    """

    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            preds = model(x)

        N = x.shape[0]
        bboxes = [[] for _ in range(N)]

        for i in range(3):
            S = preds[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_S = cells_to_bboxes(
                preds[i], anchor, S, is_pred=True,
            )

            for idx, box in enumerate(boxes_scale_S):
                bboxes[idx] += box

        # HANYA MENGGUNAKAN SCALE=52
        true_bboxes = cells_to_bboxes(
            y[2], anchors[2], S=52, is_pred=False,
        )   

        for idx in range(N):
            nms_bboxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                num_classes=num_classes
            )

            for nms_box in nms_bboxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if(box[1] > threshold):
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    
    model.train()
    return all_pred_boxes, all_true_boxes

def cells_to_bboxes(preds, anchors, S, is_pred):
    """
    Mengubah sel ke box, sehingga dapat diplot pada image asli

    Parameters:
        preds (tensor): kotak prediksi (N, 3, S, S, C+5)
        anchors (tensor): anchor yang digunakan untuk prediksi
        S (int): ukuran panjang dan lebar
        is_pred (bool): True jika prediksi, False jika target

    Returns:
        list: sel yang telah dibentuk ke bounding box (N, 3*S*S, 6)
    """

    N = preds.shape[0]
    num_anchors = len(anchors)
    box_predictions = preds[..., 1:5]

    if(is_pred):
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors
        scores = torch.sigmoid(preds[..., 0:1])
        best_classes = torch.argmax(preds[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = preds[..., 0:1]
        best_classes = preds[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(N, 3, S, 1)
        .unsqueeze(-1)
        .to(preds.device)
    )

    x = 1/S * (box_predictions[..., 0:1] + cell_indices)
    y = 1/S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h =1/S * (box_predictions[..., 2:4])
    converted_bboxes = torch.cat((best_classes, scores, x, y, w_h), dim=-1).reshape(N, num_anchors*S*S, 6)

    return converted_bboxes.tolist()

def plot_images(model, loader, iou_threshold, threshold, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(config.DEVICE)
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_S = cells_to_bboxes(
                out[i], anchor, S, is_pred=True,
            )
            
            for idx, box in enumerate(boxes_scale_S):
                bboxes[idx] += box
    
    model.train()
     
    for i in range(5):
        nms_bboxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_threshold, threshold=threshold,num_classes=1
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_bboxes)

def save_checkpoint(state, filename="model.pth.tar"):
    torch.save(state, filename)
    print("##### Berhasil menyimpan checkpoint #####")
    
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("##### Berhasil memuat checkpoint #####")
