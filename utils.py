import config
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import numpy as np 
import torch
from collections import Counter 
from tqdm import tqdm
from merge_sort import merge_sort, left_bound

def iou_width_height(boxes1, boxes2):
    """
    Menghitung intersection over union

    Parameters:
        boxes1 (tensor): lebar dan panjang dari kotak pertama
        boxes2 (tensor): lebar dan panjang dari kotak kedua

    Returns:
        tensor: intersection over union dari kotak pertama dan kedua
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) \
                   * torch.min(boxes1[..., 1], boxes2[..., 1])

    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] \
            - intersection

    return intersection / union

def intersection_over_union(pred_boxes, true_boxes, box_format="midpoint"):
    """
    Menghitung intersection over union dari kotak prediksi dan target

    Parameters:
        pred_boxes (tensor): kotak prediksi
        true_boxes (tensor): kotak target
        box_format (str): midpoint (x, y, w, h) atau corners (x1, y1, x2, y2)

    Returns:
        tensor: intersection over union dari kotak prediksi dan target
    """

    if(box_format == "midpoint"):
        box1_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3]/2
        box1_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4]/2
        box1_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3]/2
        box1_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4]/2
        box2_x1 = true_boxes[..., 0:1] - true_boxes[..., 2:3]/2
        box2_y1 = true_boxes[..., 1:2] - true_boxes[..., 3:4]/2
        box2_x2 = true_boxes[..., 0:1] + true_boxes[..., 2:3]/2
        box2_y2 = true_boxes[..., 1:2] + true_boxes[..., 3:4]/2

    elif(box_format == "corners"):
        box1_x1 = pred_boxes[..., 0:1]
        box1_y1 = pred_boxes[..., 1:2]
        box1_x2 = pred_boxes[..., 2:3]
        box1_y2 = pred_boxes[..., 3:4]
        box1_x1 = true_boxes[..., 0:1]
        box1_y1 = true_boxes[..., 1:2]
        box1_x2 = true_boxes[..., 2:3]
        box1_y2 = true_boxes[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_luas = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_luas = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_luas + box2_luas - intersection + 1e-6)

def non_max_suppression(boxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Mengurangi kotak prediksi dengan teknik non max suppression
    
    Parameters:
        bboxes (list): [[class_pred, prob_score, x, y, width, height]]
        iou_threshold (float): threshold dimana kotak prediksi berdekatan
        threshold (float): threshold untuk menghapus kotak prekdisi
        box_format (str): midpoint atau corners
        
    Returns:
        list: kotak prediksi setelah dikenakan non max suppression dengan iou threshold tertentu
    """

    boxes = [box for box in boxes if box[1] > threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    boxes_after_nms = []

    while boxes:
        chosen_box = boxes.pop(0)

        boxes = [
            box for box in boxes
            if(int(box[0]) != int(chosen_box[0]) 
            or intersection_over_union(
                torch.tensor(box[..., 2:]),
                torch.tensor(chosen_box[..., 2:]),
                box_format=box_format,
                ) < iou_threshold
            )
        ]

        boxes_after_nms.append(chosen_box)

    return boxes_after_nms

def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20
):
    """
    Menghitung mean average precision
    
    Parameters:
        pred_boxes (list): [[train_idx, class_pred, prob_score, x, y, width, height]]
        true_boxes (list): sama seperti pred_boxes
        iou_threshold (float): threshold dimana kotak prediksi benar
        box_format (str): midpoint atau corners
        num_classes (int): jumlah kelas
        
    Returns:
        float: nilai mAP untuk iou threshold tertentu
    """

    average_precisions = []
    epsilon = 1e-6
    
    # Mengurutkan pred_boxes and true_boxes menggunakan Merge Sort
    pred_boxes = merge_sort(pred_boxes, 0, len(pred_boxes)-1)
    true_boxes = merge_sort(true_boxes, 0, len(true_boxes)-1)
    
    for c in range(num_classes):
        "INGAT: index yang diberikan adalah lower bound"
        pred_lb = left_bound(pred_boxes, c)
        pred_rb = left_bound(pred_boxes, c+1)      
        true_lb = left_bound(true_boxes, c)
        true_rb = left_bound(true_boxes, c+1)
        
        "Jika true boxes kosong untuk kelas c,"
        "maka lanjut ke kelas selanjutnya"
        if(true_boxes[true_lb][1] != c):
            continue
        
        if(len(pred_boxes) and pred_boxes[pred_lb][1] == c):
            pred_rb = max(pred_lb, pred_rb-1)
            assert pred_boxes[pred_rb][1] == c, "Batas kanan dari kotak prediksi salah"
        if(len(true_boxes) and true_boxes[true_lb][1] == c):
            true_rb = max(true_lb, true_rb-1)
            assert true_boxes[true_rb][1] == c, "Batas kanan dari kotak target salah"
        
        temp = []
        for gt in true_boxes[true_lb:true_rb+1]:
            temp.append(int(gt[0]))
            assert(gt[1] == c)
        amount_bboxes = Counter(temp)
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        TP = torch.zeros(pred_rb - pred_lb + 1)
        FP = torch.zeros(pred_rb - pred_lb + 1)
        total_true_boxes = true_rb - true_lb + 1
        
        if(len(pred_boxes) and pred_boxes[pred_lb][1] == c):
            for detection_idx, detection in enumerate(pred_boxes[pred_lb:pred_rb+1]):
                assert(detection[1] == c)
                
                best_iou = 0.0
                best_gt_idx = -1
                idx = -1
                for gt in true_boxes[true_lb:true_rb+1]:
                    if(gt[0] != detection[0]):
                        continue
                    assert(gt[1] == c)
                    idx += 1
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:                        
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
                
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions)/len(average_precisions)

def plot_image(image, boxes):
    """
    Menampilkan kotak prediksi pada gambar
    """

    img = np.array(image)
    panjang, lebar, _ = img.shape
    labels = config.PASCAL_CLASSES
    cmap = plt.get_cmap("tab20b")
    colors = [cmap[i] for i in np.linspace(0, 1, len(labels))]

    _, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        class_pred = int(box[0])
        box = box[2:]
        x = box[0] - box[2]/2
        y = box[1] - box[3]/2

        ax.add_patch(
            patches.Rectangle(
                (x * lebar, y * panjang),
                box[2] * lebar,
                box[3] * panjang,
                linewidth=2,
                edgecolor=colors[class_pred],
                facecolor="none",
            )
        )

        plt.text(
            x * lebar,
            y * panjang,
            s=labels[class_pred],
            color="white",
            verticalalignment="top",
            bboxes={"color": colors[class_pred], "pad": 0},
        )
    
    plt.show()

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format,
    device
):
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for _, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_boxes(
                predictions[i], anchor, S=S, is_pred=True,
            )

            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Satu scale saja
        true_boxes = cells_to_boxes(
            labels[2], anchor, S=S, is_pred=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_boxes:
                if(box[1] > threshold):
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def cells_to_boxes(predictions, anchors, S, is_pred=True):
    batch_size=predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if(is_pred):
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = torch.arange(S)\
        .repeat(predictions.shape[0], 3, S, 1)\
        .unsqueeze(-1)\
        .to(predictions.device)

    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices)
    w_h = 1 / S * (box_predictions[..., 2:4])
    converted_bboxes = torch.cat([best_class, scores, x, y, w_h], dim=-1)\
        .reshape(batch_size, num_anchors * S * S, 6)
    
    return converted_bboxes.tolist()

def save_checkpoint(state, filename="model.pth.tar"):
    torch.save(state, filename)
    print("##### Berhasil menyimpan checkpoint #####")
    
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("##### Berhasil memuat checkpoint #####")
