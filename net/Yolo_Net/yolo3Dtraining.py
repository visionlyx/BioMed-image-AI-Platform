import numpy as np
import torch
import torch.nn as nn
import math
from utils.utils_detect.utils import bbox_iou


def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 3] / 2, _box_a[:, 0] + _box_a[:, 3] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b1_z1, b1_z2 = _box_a[:, 2] - _box_a[:, 3] / 2, _box_a[:, 2] + _box_a[:, 3] / 2

    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 3] / 2, _box_b[:, 0] + _box_b[:, 3] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    b2_z1, b2_z2 = _box_b[:, 2] - _box_b[:, 3] / 2, _box_b[:, 2] + _box_b[:, 3] / 2

    box_a = torch.zeros(int(_box_a.shape[0]), 6)
    box_b = torch.zeros(int(_box_b.shape[0]), 6)

    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3], box_a[:, 4], box_a[:, 5] = b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3], box_b[:, 4], box_b[:, 5] = b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2

    A = box_a.size(0)
    B = box_b.size(0)

    max_xyz = torch.min(box_a[:, 3:].unsqueeze(1).expand(A, B, 3), box_b[:, 3:].unsqueeze(0).expand(A, B, 3))
    min_xyz = torch.max(box_a[:, :3].unsqueeze(1).expand(A, B, 3), box_b[:, :3].unsqueeze(0).expand(A, B, 3))

    inter = torch.clamp((max_xyz - min_xyz), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1] * inter[:, :, 2]

    area_a = ((box_a[:, 3] - box_a[:, 0]) * (box_a[:, 4] - box_a[:, 1]) * (box_a[:, 5] - box_a[:, 2])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 3] - box_b[:, 0]) * (box_b[:, 4] - box_b[:, 1]) * (box_b[:, 5] - box_b[:, 2])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    return inter / union


def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    return (pred - target) ** 2


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0] // 8]
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xyz = 1
        self.lambda_length = 1.0
        self.lambda_conf = 1
        self.lambda_cls = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_d = input.size(2)
        in_h = input.size(3)
        in_w = input.size(4)

        stride_d = self.img_size[2] / in_d
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h, a_d / stride_d) for a_w, a_h, a_d in self.anchors]
        prediction = input.view(bs, int(self.num_anchors / 1), self.bbox_attrs, in_d, in_h, in_w).permute(0, 1, 3, 4, 5, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])

        l = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        mask, noobj_mask, tx, ty, tz, tl, tconf, tcls, box_loss_scale_x, box_loss_scale_y, box_loss_scale_z = \
            self.get_target(targets, scaled_anchors, in_w, in_h, in_d)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tz, tl = tx.cuda(), ty.cuda(), tz.cuda(), tl.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

        loss_x = torch.sum(MSELoss(x, tx) / bs * mask)
        loss_y = torch.sum(MSELoss(y, ty) / bs * mask)
        loss_z = torch.sum(MSELoss(z, tz) / bs * mask)
        loss_l = torch.sum(MSELoss(l, tl) / bs * mask)

        conf1 = torch.sum(BCELoss(conf, tconf) * mask / bs)
        conf2 = torch.sum(BCELoss(conf, tconf) * noobj_mask / bs) *0.015
        loss_conf = conf1 + conf2
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / bs)

        loss = loss_x * self.lambda_xyz + loss_y * self.lambda_xyz + \
               loss_z * self.lambda_xyz + loss_l * self.lambda_length + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        return loss, loss_x.item(), loss_y.item(), loss_z.item(), \
               loss_l.item(), loss_conf.item(), loss_cls.item()

    def get_target(self, target, anchors, in_w, in_h, in_d):
        bs = len(target)
        anchor_index = [[0, 1, 2]][self.feature_length.index(in_w)]
        subtract_index = [0, 3][self.feature_length.index(in_w)]

        mask = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        tz = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        tl = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)
        box_loss_scale_z = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False)

        for b in range(bs):
            for t in range(target[b].shape[0]):
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                gz = target[b][t, 2] * in_d
                gl = target[b][t, 3] * in_w

                gi = int(gx)
                gj = int(gy)
                gk = int(gz)

                gw = (gx+gl)-(gx-gl)
                gh = (gy+gl)-(gy-gl)
                gd = (gz+gl)-(gz-gl)

                gt_box = torch.FloatTensor([0, 0, 0, gw, gh, gd]).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 3)), np.array(anchors)), 1))
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                best_n = np.argmax(anch_ious)

                anch_ious_current = anch_ious[subtract_index:subtract_index + 3]
                noobj_mask[b, anch_ious_current > 0.5, gk, gj, gi] = 0

                if best_n not in anchor_index:
                    continue

                if (gj < in_h) and (gi < in_w) and (gk < in_d):
                    best_n = best_n - subtract_index
                    noobj_mask[b, best_n, gk, gj, gi] = 0
                    mask[b, best_n, gk, gj, gi] = torch.pow((gl/in_w*self.img_size[0] - 4), 2) / 2 + 1

                    tx[b, best_n, gk, gj, gi] = gx - gi
                    ty[b, best_n, gk, gj, gi] = gy - gj
                    tz[b, best_n, gk, gj, gi] = gz - gk

                    tl[b, best_n, gk, gj, gi] = math.log((gw+gh+gd) / 3 / anchors[best_n + subtract_index][0])
                    box_loss_scale_x[b, best_n, gk, gj, gi] = target[b][t, 3] * 2
                    box_loss_scale_y[b, best_n, gk, gj, gi] = target[b][t, 3] * 2
                    box_loss_scale_z[b, best_n, gk, gj, gi] = target[b][t, 3] * 2

                    tconf[b, best_n, gk, gj, gi] = 1
                    tcls[b, best_n, gk, gj, gi, int(target[b][t, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tz, tl, tconf, tcls, box_loss_scale_x, box_loss_scale_y, box_loss_scale_z

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, in_d, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])
        l = prediction[..., 3]

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        grid_y = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        grid_z = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        for b_index in range(bs):
            for anc_index in range(int(self.num_anchors / 1)):
                for z_index in range(int(in_d)):
                    for y_index in range(int(in_h)):
                        for x_index in range(int(in_w)):
                            grid_x[b_index][anc_index][z_index][y_index][x_index] = x_index
                            grid_y[b_index][anc_index][z_index][y_index][x_index] = y_index
                            grid_z[b_index][anc_index][z_index][y_index][x_index] = z_index

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_lenght = torch.zeros(bs, int(self.num_anchors / 1), in_d, in_h, in_w, requires_grad=False).cuda()
        for b_index in range(bs):
            for anc_index in range(int(self.num_anchors / 1)):
                for z_index in  range(int(in_d)):
                    for y_index in range(int(in_h)):
                        for x_index in range(int(in_w)):
                            anchor_lenght[b_index][anc_index][z_index][y_index][x_index] = anchor_w[anc_index][0]

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = z.data + grid_z
        pred_boxes[..., 3] = torch.exp(l.data) * anchor_lenght

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gz = target[i][:, 2:3] * in_d
                gl = target[i][:, 3:4] * in_w

                gt_box = torch.cat((gx, gy, gz, gl), dim=1)
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)

                for t in range(target[i].shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:4])
                    noobj_mask[i][anch_iou > self.ignore_threshold] = 0

        return noobj_mask
