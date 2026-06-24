import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, a, p, n):
        return self.loss_fn(a, p, n)