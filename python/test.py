import torch
import torch.nn as nn
from torch import Tensor
import math

'''import torch
import torch.nn as nn

import torch

norm = torch.nn.LayerNorm(3)

tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape((2, 3))

out = norm(tensor)

print(out)'''

'''n_channel = 6
norm = nn.LayerNorm(10)

height = 10
width = 10
n_elements = height * width * n_channel

t = torch.arange(0, n_elements, dtype=torch.float32).mul_(10.0 / n_elements).sin().reshape(1, n_channel, height, width)

out = norm(t)
print(out)'''

def timestep_embedding(timesteps, dim, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * torch.arange(half) / half).exp()
  args = timesteps * freqs
  return torch.cat( (args.cos(), args.sin()) ).reshape(1, -1)

timesteps = Tensor([1, 2, 3]).reshape((3, 1))
dim = 10
res = timestep_embedding(timesteps, dim)

print(res)

'''n_group = 3
n_channel = 6
norm = nn.GroupNorm(n_group, n_channel)

height = 10
width = 10 
n_elements = height * width * n_channel

t = torch.arange(0, n_elements, dtype=torch.float32).mul_(10.0 / n_elements).sin().reshape(1, n_channel, height, width)

out = norm(t)
print(out.flatten())'''