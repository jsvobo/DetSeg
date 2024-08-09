import torch #needs torch installed, first problem might arise here?
from torch.utils.cpp_extension import CUDA_HOME

def test_cuda():
    print(torch.cuda.is_available()) #Should be True, IF CUDA_VISIBLE_DEVICES is set
    print(CUDA_HOME)#If toolkit is ready: the second is not None

if __name__ == "__main__": 
    test_cuda()