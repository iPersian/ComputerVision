import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, x, target):
        intersection = (x * target).sum((2, 3))
        union = x.sum((2, 3)) + target.sum((2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()