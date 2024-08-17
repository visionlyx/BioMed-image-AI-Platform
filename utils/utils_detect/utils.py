import torch
import numpy as np
import math


def bbox_iou(box1, box2, x1y1z1x2y2z2=True):
    if not x1y1z1x2y2z2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 3] / 2, box1[:, 0] + box1[:, 3] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 4] / 2, box1[:, 1] + box1[:, 4] / 2
        b1_z1, b1_z2 = box1[:, 2] - box1[:, 5] / 2, box1[:, 2] + box1[:, 5] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 3] / 2, box2[:, 0] + box2[:, 3] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 4] / 2, box2[:, 1] + box2[:, 4] / 2
        b2_z1, b2_z2 = box2[:, 2] - box2[:, 5] / 2, box2[:, 2] + box2[:, 5] / 2
    else:
        b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3], box1[:, 4], box1[:, 5]
        b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3], box2[:, 4], box2[:, 5]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0) * torch.clamp(inter_rect_z2 - inter_rect_z1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


Config = {
    "yolo": {
        "anchors": [[[4, 4, 4], [8, 8, 8], [12, 12, 12]]],
        "classes": 1,
    },
    "img_h": 160,
    "img_w": 160,
    "img_d": 160,
}


def distance(position1, position2):
    distance = math.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2 + (position1[2] - position2[2]) ** 2)
    return distance


def open_swcs2numpy(file_list):
    swc_list = []
    for i in range(len(file_list)):
        data = np.loadtxt(file_list[i])

        if(data.ndim==1 & len(data)!=0):
            data = data[None]
        swc_list.append(data)
    return swc_list


def computing_node_nums(swc_truth, swc_predict):
    acc_nodes = 0
    truth_nodes = len(swc_truth)
    predict_nodes = len(swc_predict)

    for i in range(len(swc_truth)):
        postion_t = swc_truth[i][2:5]
        for j in range(len(swc_predict)):
            postion_p = swc_predict[j][2:5]
            dis = distance(postion_t, postion_p)

            if(dis < 10):
                acc_nodes = acc_nodes + 1
                swc_predict[j][2] = -100
                swc_predict[j][3] = -100
                swc_predict[j][4] = -100
                break
    return acc_nodes, predict_nodes, truth_nodes