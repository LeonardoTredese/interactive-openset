import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F


def forward_pass(model, x, y):
    return model(x)

def test(model, dl, accuracy_fn, device):
    running_acc, scores = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = forward_pass(model, x, y)
        running_acc += [accuracy_fn(logits, y).item()]
        scores += [logits]
    return np.mean(running_acc), torch.cat(scores, 0)

