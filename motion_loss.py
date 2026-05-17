import torch
from torch import nn


class LossFunction(nn.Module):
    def __init__(self, loss1_coeff=1, loss2_coeff=1, loss3_coeff=1, loss4_coeff=0.5):
        super().__init__()
        self.loss1_coeff = loss1_coeff
        self.loss2_coeff = loss2_coeff
        self.loss3_coeff = loss3_coeff
        self.loss4_coeff = loss4_coeff

    def xywh_to_tlbr(self, bb: torch.Tensor):
        x, y, w, h = bb.unbind(-1)
        t = x - w / 2
        l = y - h / 2
        b = x + w / 2
        r = y + h / 2
        return torch.stack([t, l, b, r], dim=-1)

    def iou(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor):
        target_boxes = self.xywh_to_tlbr(target_boxes.clone())
        pred_boxes = self.xywh_to_tlbr(pred_boxes.clone())
        bb1 = target_boxes.unsqueeze(2)
        bb2 = pred_boxes.unsqueeze(1)
        xx1 = torch.maximum(bb1[..., 0], bb2[..., 0])
        yy1 = torch.maximum(bb1[..., 1], bb2[..., 1])
        xx2 = torch.minimum(bb1[..., 2], bb2[..., 2])
        yy2 = torch.minimum(bb1[..., 3], bb2[..., 3])
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        area1 = (bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])
        area2 = (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1])
        union = area1 + area2 - inter
        ious = inter / (union + 1e-7)
        return ious

    def ciou(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor):
        target_x, target_y, target_w, target_h = target_boxes.unbind(-1)
        pred_x, pred_y, pred_w, pred_h = pred_boxes.unbind(-1)
        target_tlbr = self.xywh_to_tlbr(target_boxes)
        pred_tlbr = self.xywh_to_tlbr(pred_boxes)
        t1, l1, b1, r1 = target_tlbr.unbind(-1)
        t2, l2, b2, r2 = pred_tlbr.unbind(-1)
        xx1 = torch.maximum(t1, t2)
        yy1 = torch.maximum(l1, l2)
        xx2 = torch.minimum(b1, b2)
        yy2 = torch.minimum(r1, r2)
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        area1 = (b1 - t1) * (r1 - l1)
        area2 = (b2 - t2) * (r2 - l2)
        union = area1 + area2 - inter
        iou = inter / (union + 1e-7)
        center_distance_sq = (target_x - pred_x) ** 2 + (target_y - pred_y) ** 2
        enclose_t = torch.minimum(t1, t2)
        enclose_l = torch.minimum(l1, l2)
        enclose_b = torch.maximum(b1, b2)
        enclose_r = torch.maximum(r1, r2)
        enclose_w = enclose_b - enclose_t
        enclose_h = enclose_r - enclose_l
        enclose_diagonal_sq = enclose_w ** 2 + enclose_h ** 2 + 1e-7
        distance_penalty = center_distance_sq / enclose_diagonal_sq
        arctan_target = torch.atan(target_w / (target_h + 1e-7))
        arctan_pred = torch.atan(pred_w / (pred_h + 1e-7))
        v = (4 / (torch.pi ** 2)) * torch.pow(arctan_target - arctan_pred, 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - distance_penalty - alpha * v
        return ciou

    def forward(self, preds, targets):
        pred_boxes = preds[:, :, :4]
        target_boxes = targets[:, :, :4]
        cious = self.ciou(target_boxes, pred_boxes)
        loss1 = (1 - cious).mean()
        loss2 = nn.functional.smooth_l1_loss(pred_boxes, target_boxes)
        target_cs = targets[:, :, 4]
        pred_cs = preds[:, :, 4]
        loss3 = nn.functional.smooth_l1_loss(pred_cs, cious.detach())
        return self.loss1_coeff * loss1 + self.loss2_coeff * loss2 + self.loss3_coeff * loss3
