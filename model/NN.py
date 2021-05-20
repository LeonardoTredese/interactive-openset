import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import flatten

class NN(nn.Module):
  def __init__(self, last_layer_size=10, lr=8e-2):
    super(NN, self).__init__()
    self.lr = lr
    self.activations = nn.ReLU()
    self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
    self.conv2 = nn.Conv2d(100, 200, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(3200, 50)
    self.fc2 = nn.Linear(50, last_layer_size)
    self._compile()

  def _compile(self, lr=1e-3):
      self.loss = nn.CrossEntropyLoss()
      self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
  
  def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 3200)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x, dim= -1)
