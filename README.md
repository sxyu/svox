# Sparse Voxels - PyTorch CUDA Extension

Implements a differentiable N^3 Tree:
```python
import torch
from svox import N3Tree
# N is spatial branching factor in N^3 tree
# data_dim is number of dimensions in stored data (e.g. 4 for rgba)
tree = N3Tree(N=4, data_dim=4).cuda()

# coords (Q, 3) - 3d coordinates in [0, 1), will be clamped
result = tree(coords)
# result (Q, data_dim=4)

result.backward()  # Autograd to tree nodes (tree.data)
# Can be used with optimizer etc

# Initial voxel grid is NxN (very small).
# To modify the voxel structure, use 'refine'
# Which creates a sub-grid at current leaf nodes

# Refine all leaves with value >= thresh at dimension 'dim'
# Useful for light field.
# Refines at most max_refines leaves. If exceeding,
# takes random w.p. propto (1/N^3)^{depth}
resized = result.refine_thresh(dim, thresh, max_refines)

# Refine all leafs
resized = result.refine_all()
```

**Major caveat:** after refining, the tree's parameter may be resized
(above, if the return value `resized = True`).
In such cases, you *must* manually re-create any optimizer
to `tree.parameters()` for now.
This is tricky because PyTorch assumes parameters are fixed size.

## Installation
`pip install .`

## Troubleshooting
If you get SIGSEGV upon importing,
check that your CUDA runtime and PyTorch CUDA versions match.  That is,
`nvcc --version`
should match (Python)
`torch.version.cuda`
