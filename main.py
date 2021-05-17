#!/usr/bin/python3
from model.train import train
from model.test import test
from model.NN import NN
import torch as to
import torchvision as tv
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from functools import reduce
from itertools import islice
import json

def closed_set_accuracy(logits, labels):
    _, predictions = to.max(logits, -1)
    return to.where(labels == predictions, 
                     to.ones_like(predictions), 
                     to.zeros_like(predictions)).sum() / labels.shape[0]



def filter_classes(dataset, classes):
  idx = reduce(lambda x, y: x | y,map(lambda class_: dataset.targets == class_, classes))
  return dataset.targets[idx], dataset.data[idx]

device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
models = {}
mode = "save"

if (mode == "save"):
  for i in range(2,11):
    train_classes = range(i)
    ds_train = tv.datasets.MNIST(root='./', train=True, transform=tv.transforms.ToTensor(), download=True)
    ds_train.targets, ds_train.data = filter_classes(ds_train, train_classes)
    dl_train = to.utils.data.DataLoader(ds_train, batch_size=256, shuffle=True)
    input_dim = np.prod(ds_train.__getitem__(0)[0].shape)
    args = []
    kwargs = {
        'last_layer_size': i,
        'lr': 5e-4,
    }
    model = NN(*args,**kwargs).to(to.device(device))
    epochs = 10
    print("Training model %d for %d epochs..." % (i, epochs))
    loss, acc, scores = train(model, dl_train, closed_set_accuracy, device, epochs = epochs)
    print("Trained with loss %f and accuracy  %f" % (loss[-1], acc[-1]))
    model_name = "model_%d_" % i 
    models[model_name + "state_dict"] = model.state_dict()
    models[model_name + "args"] = args
    models[model_name + "kwargs"] = kwargs
  to.save(models, './saves')

if (mode == "load"):
  models = to.load('./models', map_location=device)
  for i in range(2,11):
    test_classes = range(i)
    ds_test = tv.datasets.MNIST(root='./', train=False, transform=tv.transforms.ToTensor(), download=True)
    ds_test.targets, ds_test.data = filter_classes(ds_test, test_classes)
    dl_test = to.utils.data.DataLoader(ds_test, batch_size=256)
    model_name = "model_%d_" %i 
    args = models[model_name + "args"]
    kwargs = models[model_name + "kwargs"]
    model = NN(*args,**kwargs).to(device)
    model.load_state_dict(models[model_name + "state_dict"])
    print("Testing model %d..." % i)
    acc, _ = test(model,dl_test, closed_set_accuracy, device)
    print("Testing accuracy  %f" % acc)

if (mode == "sample"):
  batch_size = 1
  train_classes = range(9)
  ds = tv.datasets.MNIST(root='./', train=True, transform=tv.transforms.ToTensor(), download=True)
  ds.targets, ds.data = filter_classes(ds, train_classes)
  dl = to.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
  for x, y in islice(dl, 1):
    x = x.view(batch_size, -1).cpu().numpy()
    y = y.cpu().numpy()
    print(json.dumps(x[0].tolist()))