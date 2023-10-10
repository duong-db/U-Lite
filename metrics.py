import torch
import torch.nn as nn
import numpy as np

def iou_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)   

    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum()
    union = (y_pred | y_true).sum()
  
    return intersection / union

def dice_score(y_pred, y_true, smooth=0.):

    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()

    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dice = dice_score(y_pred, y_true, smooth=1e-3)

        return 1 - dice
