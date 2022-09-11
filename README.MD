# Two-Level Kronecker Product Factorization Bilinear Pooling for PyTorch.

This repository has a Python implementation of Two-Level Kronecker Product Factorization Bilinear Pooling for PyTorch by Yu et al, 2022  [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20225)].

This implementation uses `tensorly` for tensor folding and tensor modal product calculations.

`pip install -U tensorly`

## Usage

`class TKPFBilinearPooling(input_dim, a=64, b=64, r=16, q=2, cuda=True)`

Basic usage:
```
from TKPFBilinearPooling import TKPFBilinearPooling

w, h, channels = 7, 7, 2048
bp = TKPFBilinearPooling(channels, cuda=False)
x = torch.rand(4,channels, w, h)

z = bp(x)
```

## References

 - Yu Tan et al. "[Efficient Bilinear Pooling via Kronecker Product](https://ojs.aaai.org/index.php/AAAI/article/view/20225)", 2022
