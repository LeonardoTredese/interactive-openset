import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F


def forward_pass(model, x, y):
    batch_size = x.shape[0]
    model.optimizer.zero_grad()
    x = x.view(batch_size, -1)
    return model(x)

def test(model, dl, accuracy_fn, device):
    running_acc, scores = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = forward_pass(model, x, y)
        running_acc += [accuracy_fn(F.softmax(logits, -1), y).item()]
        scores += [logits]
    return np.mean(running_acc), torch.cat(scores, 0)

