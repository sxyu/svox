# PlenOctrees: PyTorch CUDA Extension

This repository contains a differentiable PlenOctree and renderer implementation
as a PyTorch CUDA extension. It is used by our conversion and optimization code.

PlenOctrees for Real Time Rendering of Neural Radiance Fields<br>
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

https://alexyu.net/plenoctrees

Please also refer to the following repositories

- NeRF-SH training and PlenOctree extraction: <https://github.com/sxyu/plenoctree>
- C++ volume renderer <https://github.com/sxyu/volrend>

## Installation
`pip install svox`

## Documentation
Please see <https://svox.readthedocs.io>

## Troubleshooting
If you get SIGSEGV upon importing,
check that your CUDA runtime and PyTorch CUDA versions match.  That is,
`nvcc --version`
should match (Python)
`torch.version.cuda`

## Misc
SVOX stands for **s**parse **v**oxel **o**ctree e**x**tension.
