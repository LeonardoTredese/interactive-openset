#!/usr/bin/python3
from train import train
from test import test
from MLP import MLP
import torch as to
import torchvision as tv
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from functools import reduce

def closed_set_accuracy(logits, labels):
    _, predictions = to.max(logits, -1)
    return to.where(labels == predictions, 
                     to.ones_like(predictions), 
                     to.zeros_like(predictions)).sum() / labels.shape[0]



def filter_classes(dataset, classes):
  idx = reduce(lambda x, y: x | y,map(lambda class_: dataset.targets == class_, classes))
  return dataset.targets[idx], dataset.data[idx]

device = to.device("cpu")
models = {}
mode = "load"
if (mode == "save"):
  for i in range(1,10):
    train_classes = range(i)
    ds_train = tv.datasets.MNIST(root='./', train=True, transform=tv.transforms.ToTensor(), download=True)
    ds_train.targets, ds_train.data = filter_classes(ds_train, train_classes)
    dl_train = to.utils.data.DataLoader(ds_train, batch_size=256, shuffle=True)
    input_dim = np.prod(ds_train.__getitem__(0)[0].shape)
    args = [input_dim]
    kwargs = {
        'layers_size': [20, i],
        'lr': 5e-4,
        'dropout_rate': 0.2
    }
    model = MLP(*args,**kwargs).to(to.device('cpu'))
    epochs = 10
    print("Training model %d for %d epochs..." % (i, epochs))
    loss, acc, scores = train(model, dl_train, closed_set_accuracy, device, epochs = epochs)
    print("Trained with loss %f and accuracy  %f" % (loss[-1], acc[-1]))
    model_name = "model_%d_" % i 
    models[model_name + "state_dict"] = model.state_dict()
    models[model_name + "args"] = args
    models[model_name + "kwargs"] = kwargs
  to.save(models, './models')

if (mode == "load"):
  models = to.load('./models')
  for i in range(1,10):
    test_classes = range(i)
    ds_test = tv.datasets.MNIST(root='./', train=False, transform=tv.transforms.ToTensor(), download=True)
    ds_test.targets, ds_test.data = filter_classes(ds_test, test_classes)
    dl_test = to.utils.data.DataLoader(ds_test, batch_size=256)
    model_name = "model_%d_" %i 
    args = models[model_name + "args"]
    kwargs = models[model_name + "kwargs"]
    model = MLP(*args,**kwargs).to(device)
    model.load_state_dict(models[model_name + "state_dict"])
    print("Testing model %d..." % i)
    acc, _ = test(model,dl_test, closed_set_accuracy, device)
    print("Testing accuracy  %f" % acc)

