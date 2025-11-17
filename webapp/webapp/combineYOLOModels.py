#combineYOLOModels.py
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


def ensemble_predictions(results1, results2, iou_threshold=0.5):
    """ Kombiniert zwei YOLO-Ergebnisse durch Non-Maximum Suppression (NMS).
        Wählt immer die Klasse mit der höchsten Confidence.
    """
    boxes1 = results1[0].boxes.xyxy.cpu().numpy()
    scores1 = results1[0].boxes.conf.cpu().numpy()
    classes1 = results1[0].boxes.cls.cpu().numpy().astype(int)

    boxes2 = results2[0].boxes.xyxy.cpu().numpy()
    scores2 = results2[0].boxes.conf.cpu().numpy()
    classes2 = results2[0].boxes.cls.cpu().numpy().astype(int)

    # Ergebnisse zusammenführen
    boxes = np.vstack((boxes1, boxes2))
    scores = np.hstack((scores1, scores2))
    classes = np.hstack((classes1, classes2))

    # Non-Maximum Suppression (NMS)
    indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

    # Korrekte Auswahl der finalen Bounding Boxes, Scores und Klassen
    final_boxes = boxes[indices.numpy()]
    final_scores = scores[indices.numpy()]
    final_classes = classes[indices.numpy()]

    return final_boxes, final_scores, final_classes
