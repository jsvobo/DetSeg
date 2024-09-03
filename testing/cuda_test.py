import torch  # needs torch installed, first problem might arise here?
from torch.utils.cpp_extension import CUDA_HOME


def test_cuda():
    assert torch.cuda.is_available()
    assert CUDA_HOME is not None
    print(CUDA_HOME)  # If toolkit is ready: the second is not None


if __name__ == "__main__":
    test_cuda()
