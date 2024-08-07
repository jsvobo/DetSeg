import torch
print(torch.cuda.is_available())
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)