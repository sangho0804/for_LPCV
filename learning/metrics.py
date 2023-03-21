import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)
    



#need to fix
class mIoU(Metric):
    def __init__(self, num_classes, eps=1e-7):
        super(mIoU, self).__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.iou_scores = torch.zeros(num_classes)

    def forward(self, y_pred, y_true):
        #print(y_pred.shape)
        #print(y_true.shape)
        y_pred = F.softmax(y_pred, dim=1)
        #print(" max y_pred",y_pred.shape)
        #y_true = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2)
        #print(y_true.shape)
        #print(y_pred[:, 0, :, :])
        iou_scores = []
        for i in range(self.num_classes):
            intersection = torch.sum(y_pred[:, i, :, :] * y_true[:, i, :, :])
            union = torch.sum(y_pred[:, i, :, :]) + torch.sum(y_true[:, i, :, :]) - intersection
            iou_i = (intersection + self.eps) / (union + self.eps)
            iou_scores.append(iou_i.item())
        self.iou_scores = torch.tensor(iou_scores)
        miou = torch.mean(self.iou_scores)
        return miou
    
    
#need to fix
class DiceScore(Metric):
    def __init__(self, num_classes, eps=1e-7):
        super(DiceScore, self).__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.iou_scores = torch.zeros(num_classes)

    def forward(self, y_pred, y_true):
        #print(y_pred.shape)
        #print(y_true.shape)
        y_pred = F.softmax(y_pred, dim=1)
        #print(" max y_pred",y_pred.shape)
        #y_true = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2)
        #print(y_true.shape)
        dice_scores = []
        #print(y_true[:,0,:,:])
        for i in range(self.num_classes):
            intersection = torch.sum(y_pred[:, i, :, :] * y_true[:, i, :, :])
            union = torch.sum(y_pred[:, i, :, :]) + torch.sum(y_true[:, i, :, :]) + intersection
            dice = (2 * intersection + self.eps) / (union + self.eps)
            dice_scores.append(dice.item())
        self.dice_scores = torch.tensor(dice_scores)
        mdice = torch.mean(self.dice_scores)
        return mdice   