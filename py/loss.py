import torch.nn as nn
import torch

class SmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, dim=-1):
        super().__init__()

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CrossEntropy(nn.Module):
  def init(self):
    super().__Init__()

  def forward(self, x, y):
    return ((x * -torch.log(y+1e-5)).sum(dim=1)).mean()