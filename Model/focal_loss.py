import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



# Accuracy car brand: 0.2665709008824133
# Accuracy car model: 0.04247896572953006
# Accuracy body: 0.24379232505643342
# Accuracy color: 0.22593884670634107
# Accuracy engine type: 0.22717012107531295
# Accuracy transmission: 0.37348655858813873
# Roc auc score rudder: 0.49723037807425835

# Accuracy car brand: 0.9437718038169506
# Accuracy car model: 0.8214652164990766
# Accuracy body: 0.8586086599630618
# Accuracy color: 0.8251590396059922
# Accuracy engine type: 0.9408988302893495
# Accuracy transmission: 0.7775497640057459
# Roc auc score rudder: 0.6520735485072594

# Accuracy car brand: 0.95752103427047
# Accuracy car model: 0.8504001641699158
# Accuracy body: 0.8727683152062384
# Accuracy color: 0.8440385799302278
# Accuracy engine type: 0.9437718038169506
# Accuracy transmission: 0.7847321978247486
# Roc auc score rudder: 0.6564627732942556