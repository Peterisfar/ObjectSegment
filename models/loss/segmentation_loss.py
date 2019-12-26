import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, reduction="mean", ignore_index=255):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)

        loss = criterion(logit, target.long())

        return loss