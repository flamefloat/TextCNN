import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

inputs = torch.randn(1, 5, 2)
print(inputs)
m = nn.MaxPool2d((3, 2))
outputs = m(inputs)
print(outputs)
a = torch.tensor([[1,2,3],[4,5,6]])
print('a:', a, a.size())
b = a.unsqueeze(0)
print('b:', b, b.size())
inputss = torch.randn(5, 1)
ma = inputss.data.max(0, keepdim=True)
print(inputss, ma)

