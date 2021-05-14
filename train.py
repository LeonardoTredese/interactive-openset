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

def backward_pass(model, logits, y):
    loss_batch = model.loss(logits, y)
    loss_batch.backward()
    model.optimizer.step()
    return loss_batch.item()

def run_epoch(model, dl, accuracy_fn, device):
    running_loss, running_acc, scores = [], [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = forward_pass(model, x, y)
        running_acc += [accuracy_fn(F.softmax(logits, -1), y).item()]
        scores += [logits]
        running_loss += [backward_pass(model, logits, y)]
    return np.mean(running_loss), np.mean(running_acc), torch.cat(scores, 0)


def train(model, dl, accuracy_fn, device, epochs=10):
  model.train()
  loss, acc, scores = [], [], []
  for epochs in range(epochs):
    loss_epoch, acc_epoch, scores_epoch = run_epoch(model, dl, accuracy_fn, device)
    loss += [loss_epoch]
    acc += [acc_epoch]
    scores += [scores_epoch]
  return loss, acc, scores
