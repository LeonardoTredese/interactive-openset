import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
  def __init__(self, in_dim, layers_size=[10], dropout_rate=0.0, lr=8e-2):
    super(MLP, self).__init__()
    self.in_dim = in_dim
    self.lr = lr
    self.dropout = nn.Dropout(dropout_rate)
    self.activations = nn.ReLU()
    self.layers = self._get_layers(layers_size)
    self._compile()

  def _get_layers(self, layers_size):
    layers = []
    in_dim = self.in_dim
    for idx_l, l_size in enumerate(layers_size):
      layers += [nn.Linear(in_dim, l_size)]
      if idx_l < len(layers_size) - 1:
        layers += [self.activations]
        layers += [self.dropout]
      in_dim = l_size
    return nn.Sequential(*layers)

  def _compile(self, lr=1e-3):
      self.loss = nn.CrossEntropyLoss()
      self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, x):
      x = self.layers(x)
      return x

