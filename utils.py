import torch
from torch import nn

def compute_source_loss(logits_weak, logits_strong, labels):
    """
    Receives logits as input (dense layer outputs with no activation function)
    """
    loss_function = nn.CrossEntropyLoss() # default: `reduction="mean"`
    weak_loss = loss_function(logits_weak, labels)
    strong_loss = loss_function(logits_strong, labels)

    return weak_loss + strong_loss


def compute_target_loss(pseudolabels, logits_strong, mask):
    """
    Receives logits as input (dense layer outputs with no activation function).
    `pseudolabels` are treated as ground truth (standard SSL practice).
    """
    loss_function = nn.CrossEntropyLoss(reduction="none")
    pseudolabels = pseudolabels.detach() # remove from backpropagation

    loss = loss_function(logits_strong, pseudolabels)
    loss *= mask
    
    return torch.mean(loss, 0)