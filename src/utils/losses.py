import math
import torch
import torch.nn as nn
from . import lovasz_losses
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class OHEMLoss(nn.Module):
    # https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
    # TODO: Smoothing, Focal Loss
    def __init__(self, rate, smooth):
        super(OHEMLoss, self).__init__()
        self.rate = rate
        self.smooth = smooth
        self.criterion = SmoothCrossEntropyLoss(reduction="none", smoothing=self.smooth)
        
    def forward(self, input, target):
        batch_size = input.size(0) 
        ohem_cls_loss = self.criterion.forward(input, target)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*self.rate) )
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        loss = ohem_cls_loss.sum() / keep_num
        return loss

class SmoothCrossEntropyLoss(_WeightedLoss):
    # https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()     

        return loss

class FocalLoss(nn.Module): #https://gist.github.com/yudai09/c1ae3ef9bbf1333acc20b28189df9968#file-focal_loss_pytroch-py
    def __init__(self, alpha=1, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha  = alpha
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = F.cross_entropy(input, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = self.alpha * loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.mean()
            
class AutoFocalLoss(nn.Module): #https://arxiv.org/abs/1904.09048
    def __init__(self, alpha=1, h=0.7, eps=1e-7):
        super(AutoFocalLoss, self).__init__()
        self.h = h
        self.alpha  = alpha
        self.pc = 0
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = F.cross_entropy(input, target, reduction="none")
        view = target.size() + (1,)
        with torch.no_grad():
            self.pc = self.pc*0.95 + logit.mean(axis=1)*0.05
            k = self.h * self.pc + (1 - self.h)
            gamma = torch.log(1 - k) / torch.log(1 - self.pc) - 1
        index = target.view(*view)
        loss = self.alpha * loss * (1 - logit.gather(1, index).squeeze(1)) ** gamma # focal loss

        return loss.mean()
    
class SoftmaxLovaszLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLovaszLoss, self).__init__()
    def forward(self, logits, targets):
        return lovasz_losses.lovasz_softmax(logits, targets)

class FocalLovaszLoss(FocalLoss):
    def __init__(self, alpha=1, gamma=2, beta = 1, eps=1e-7):
        super(FocalLovaszLoss, self).__init__(alpha, gamma, eps)
        self.beta = beta
        
    def forward(self, logits, targets):
        lovasz_loss = lovasz_losses.lovasz_softmax(logits, targets)
        focal_loss = super().forward(logits, targets)
        loss = (focal_loss+self.beta*lovasz_loss)/(1+self.beta)
        return loss

    
class AdaCosLoss(nn.Module):
    def __init__(self, num_classes, m=0.50, beta = 1, classify_loss=None):
        super(AdaCosLoss, self).__init__()
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.beta = beta
        self.classify_loss1 = nn.CrossEntropyLoss()
        if classify_loss is None:
            self.classify_loss2 = nn.CrossEntropyLoss()
        else:
            self.classify_loss2 = classify_loss()

    def forward(self, logits, labels):
        cosine = logits
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / logits.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        loss1 = self.classify_loss1(output, labels)
        loss2 = self.classify_loss2(cosine, labels)
        loss=(self.beta*loss1+ loss2)/(1+self.beta)
        return loss

def mean(x:tuple):
    return float(sum(x)) / len(x)
