import torch.nn as nn

class MASE(nn.Module):
    def __init__(self):
        super(MASE, self).__init__()

    def forward(self, predictions, target):
        mae = (predictions - target).abs().mean()
        maenaive = 0

        for i in range(1,target.size()[1]):
            maenaive += (target[:, i, :] - target[:, i-1, :]).abs().mean()
        maenaive = maenaive.sum()/(target.size()[1] -1)
        mase = mae/maenaive

        return mase
    

class SMAPE(nn.Module):
    def __init__(self, eps=1e-8):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, predictions, target):
        abs_err = (predictions - target).abs()
        m = (predictions.abs() + target.abs() + self.eps)/2
        smape = (abs_err/m).mean()

        return smape