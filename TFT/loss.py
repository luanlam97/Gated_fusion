import torch
from torch import nn
import torch.nn.functional as F


class QuantilesLoss(nn.Module):
    def __init__(self, device= 'cpu',quantiles_list = [.1, .5, .9]):
        super().__init__()
        self.quantiles_list = torch.tensor(quantiles_list, device = device)

    def forward(self, predicted, targets):
        loss = (1 - self.quantiles_list) * F.relu(predicted - targets) +  self.quantiles_list * F.relu(targets - predicted)
        loss = loss.mean(dim=(0,1))
        return loss