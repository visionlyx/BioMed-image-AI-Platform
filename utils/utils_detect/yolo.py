import numpy as np
import os
import torch
import math
import torch.nn as nn
from net.Yolo_Net.yolo3D import Yolo3DBody


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_depth = input.size(2)
        input_height = input.size(3)
        input_width = input.size(4)

        stride_d = self.img_size[2] / input_depth
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h, anchor_depth / stride_d ) for anchor_width, anchor_height, anchor_depth in self.anchors]
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs, input_depth, input_height, input_width).permute(0, 1, 3, 4, 5, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])
        l = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        grid_y = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        grid_z = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()

        for b_index in range(batch_size):
            for anc_index in range(int(self.num_anchors)):
                for z_index in range(int(input_depth)):
                    for y_index in range(int(input_height)):
                        for x_index in range(int(input_width)):
                            grid_x[b_index][anc_index][z_index][y_index][x_index] = x_index
                            grid_y[b_index][anc_index][z_index][y_index][x_index] = y_index
                            grid_z[b_index][anc_index][z_index][y_index][x_index] = z_index

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_lenght = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()


        for b_index in range(batch_size):
            for anc_index in range(int(self.num_anchors)):
                for z_index in range(int(input_depth)):
                    for y_index in range(int(input_height)):
                        for x_index in range(int(input_width)):
                            anchor_lenght[b_index][anc_index][z_index][y_index][x_index] = anchor_w[anc_index][0]

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = z.data + grid_z
        pred_boxes[..., 3] = torch.exp(l.data) * anchor_lenght

        _scale = torch.Tensor([stride_w, stride_h, stride_d, stride_w]).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data


Config = {
    "yolo": {
        "anchors": [[[4, 4, 4], [8, 8, 8], [12, 12, 12]]],
        "classes": 1,
    },
    "img_h": 160,
    "img_w": 160,
    "img_d": 160,
}


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


def non_max_suppression(prediction, num_classes, conf_thres, nms_thres):
    x1 = prediction[:, :, 0:1] - prediction[:, :, 3:4] / 2
    y1 = prediction[:, :, 1:2] - prediction[:, :, 3:4] / 2
    z1 = prediction[:, :, 2:3] - prediction[:, :, 3:4] / 2
    x2 = prediction[:, :, 0:1] + prediction[:, :, 3:4] / 2
    y2 = prediction[:, :, 1:2] + prediction[:, :, 3:4] / 2
    z2 = prediction[:, :, 2:3] + prediction[:, :, 3:4] / 2

    prediction = torch.cat((x1, y1, z1, x2, y2, z2, prediction[:, :, 4:]), -1)
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 6] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(image_pred[:, 7:7 + num_classes], 1, keepdim=True)
        detections = torch.cat((image_pred[:, :7], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(detections_class[:, 6], descending=True)
            detections_class = detections_class[conf_sort_index]
            max_detections = []

            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break

                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    return output


def distance_suppression(prediction, dis_thres):
    output = []
    x_c = (prediction[:, 3] + prediction[:, 0]) / 2
    y_c = (prediction[:, 4] + prediction[:, 1]) / 2
    z_c = (prediction[:, 5] + prediction[:, 2]) / 2
    r = (prediction[:, 3] - prediction[:, 0]) / 2

    for i in range(len(prediction)):
        temp = []
        temp.append(prediction[i])

        for j in range(len(prediction)):
            if i == j:
                continue
            d = math.sqrt((x_c[i] - x_c[j]) ** 2 + (y_c[i] - y_c[j]) ** 2 + (z_c[i] - z_c[j]) ** 2)

            if d < (r[i] + r[j]) * dis_thres or d < 5:
                temp.append(prediction[j])

        best_conf = temp[0]
        if(len(temp) > 1):
            for k in range(1, len(temp)):
                if temp[k][6] > best_conf[6]:
                    best_conf = temp[k]
        output.append(best_conf)

    resultList = []
    resultList.append(output[0])

    for i in range(1, len(output)):
        copy = 0
        for tt in resultList:
            if (tt == output[i]).all():
                copy = 1
        if copy == 0:
            resultList.append(output[i])

    resultList = np.array(resultList)
    return resultList


class YOLO(object):
    _defaults = {
        "classes_path": 'data/kidney_detect/validate/classes.txt',
        "model_image_size": (160, 160, 160, 1),
        "confidence": 0.88,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, mode, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.model_path = mode
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = Yolo3DBody(self.config)

        checkpoint = torch.load(self.model_path)
        self.net.load_state_dict(checkpoint['model'])
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(1):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"], (self.model_image_size[2], self.model_image_size[1], self.model_image_size[0])))

    def detect_image(self, images):
        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.net(images)
            outputs = outputs[np.newaxis, :]
            output_list = []

            for i in range(1):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"], conf_thres=self.confidence, nms_thres=0.001)

        if (batch_detections[0] != None):
            batch_detections = batch_detections[0].cpu().numpy()
            batch_detections = distance_suppression(batch_detections, 1.5)
            top_bboxes = np.array(batch_detections[:, :6])
            return top_bboxes
        else:
            return []

