.. _quick:

Quick Guide
==================

Below I give a quick overview of some core functionality of svox to help get you started.
Please see :ref:`svox` for detailed per-method documentation.
To install the library, simply use :code:`pip install sxox`; you would of course need
to install PyTorch first.
You will also need the CUDA runtime to compile the CUDA extension;
while the library works without the CUDA extension, it is very slow, and will emit a warning the
first time a CUDA-capable operation is used.

If the extension fails to build, check if your PyTorch is using the same CUDA
version as you have installed on your system.

Construction
-------------------------------

We begin by importing the library and constructing a tree:

>>> import svox
>>> t=svox.N3Tree(data_dim=4, data_format="RGBA",
                  center=[0.5, 0.5, 0.5], radius=0.5,
                  N=2, device="cpu",
                  init_refine=0, depth_limit=10,
                  extra_data=None)
>>> t.cuda()

* :code:`data_dim` is the size of data to store in each leaf, for example 4 for :code:`RGBA` data.
  Since 0.2.28: this is optional if :code:`data_format` is something other than :code:`RGBA`.
  Since 0.2.27: an error is thrown if this is incompatible with :code:`data_format`.
* :code:`data_format`, a bit redundant, is the data format for rendering (only used for VolumeRenderer).
  It can be :code:`RGBA`, :code:`SH#`, :code:`SG#`, or :code:`ASG#`, where # (basis_dim) is the dimensionality of the basis function.
  This is somewhat redundant with data_dim.
  data_dim should be :code:`basis_dim * 3 + 1` (Last item is always :math:`\sigma \in [0, \infty)`, the density).
  For SH (spherical harmonics), basis_dim must be a square number at most 25. SG (spherical Gaussians) and ASG (anisotropic SG) require :code:`extra_data` field to render properly.
* :code:`radius` and :code:`center` specify the transform of the tree in space, with :code:`radius` meaning the half-edge length of the bounding cube (1 float or list of 3 floats for each axis) and :code:`center` specifying the center of the cube (list of 3 floats).  By default cube is centered at :code:`[0.5, 0.5, 0.5]` with radius 0.5.
* :code:`N` (optional, default 2) is the N in :math:`N^3` tree. Typically, put :code:`N=2` for an octree.
* :code:`device` (optional, default cpu) can be a string like 'cuda' and is where the tree's data will be stored.
* :code:`init_refine` specifies initial LOD of the tree: the initial leaf voxel size will be :code:`N^(init_refine + 1)`.
* :code:`depth_limit` is a utility for limiting the maximum depth of any tree leaf after refinement.  Note that the root is at depth -1, which may be a bit confusing; initially the tree has maximum depth 1 and :code:`NxNxN` leaves.
* :code:`extra_data` for SG, basis_dim x 4 matrix of variance/mean (3). For ASG, data_dim x 11 matrix.
  Currently, optimizing wrt this matrix is not supported, so the parameters should be pre-determined.

:code:`svox.N3Tree` is a PyTorch module and
usual operations such as :code:`.parameters()` or :code:`.cuda()` can be used.
The forward method of the N3Tree class takes a batch of points :code:`(B, 3)` and returns
corresponding data.

Saving and Loading
------------------------------
To save and load trees to/from npz files, use

>>> tree.save(npz_path)
>>> tree = svox.N3Tree.load(npz_path, device=device)

'device' can be a string like 'cuda' and is where the tree's data will be loaded into, similar
to that in the constructor.
Since the tree is a PyTorch module, you could also use a PyTorch checkpoint, but it can be VERY inefficient.

Querying and Modifying Data using N3TreeView
---------------------------------------------

For convenient query and manipulation, we provide an approximate analogy to the PyTorch tensor,
where the tree is viewed as a matrix of size
:code:`(n_leaves, data_dim)`. Any indexing operation into the N3Tree returns a
:code:`N3TreeView` class which works like a tensor.

>>> tree.shape
torch.Size([8, 4])
>>> tree[0] += 1
>>> tree[:, :3]
N3TreeView(tensor([[1., 1., 1.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]))
>>> tree[:, -1:] = -1
>>> tree[:2]
N3TreeView(tensor([[ 1.,  1.,  1., -1.],
        [ 0.,  0.,  0., -1.]]))

You can also, of course, query the tree using *real spatial points*,
by either using 3 indices or a :code:`(N, 3)` query matrix:

>>> tree[0, 0.5, 0]   # Query point (0, 0.5, 0)
>>> tree[points]      # Query points (N, 3)

This returns a N3TreeView of leaves corresponding to these points, so you can also modify them:

>>> tree[points] = values      # Query points (N, 3), values (N, data_dim)
>>> tree[0, 0, 0] /= 2.0

The tree is self behaves similarly to :code:`tree[:]`. Some more examples:

>>> tree += 1.5
>>> tree.normal_()
>>> tree[:, -1:].clamp_(0.1, None)

When used with a PyTorch operation such as :code:`torch.mean` or operators like :code:`+`,
the N3TreeView is queried and the values are converted to a PyTorch tensor automatically.
If you wish to get the values as a tensor explicitly, use :code:`view.values`.
See the section :ref:`Advanced Leaf-level Accessors<leaf_level_acc>` for more advanced operations supported by
N3TreeView.

Refinement oracle
---------------------

To refine the tree, use the :code:`refine` function.
The first argument allows you to refine more than once.

>>> tree.refine()  # Refine all nodes
>>> tree.refine(2)  # Refine all nodes twice
>>> tree[-1].refine()  # Refine leaf -1 once, through the N3TreeView

Differentiable Volume Rendering
---------------------------------
This is implemented in the :code:`svox.VolumeRenderer` class.
The following code renders a perspective image:

>>> ren = svox.VolumeRenderer(tree)
>>> camera = # some [4, 4] camera pose matrix
>>> ren.render_persp(camera, width=width, height=height, fx=fx) # Get a perspective image

Note the renderer need not be updated if the tree is modified.
The renderer will use the tree's :code:`data_format` field:
one of :code:`RGBA`, :code:`SH#`, :code:`SG#`, or :code:`ASG#`, where # (basis_dim) is the dimensionality of the basis function.
For SH, this must be a square number at most 25.
The last dimension is always used as density :math:`\sigma \in [0, \infty)`, where the value is clipped to 0 while rendering if negative.
The volume rendering formula is as in NeRF:

.. math::
    \mathbf{C} = \sum_{i=1}^n \left[\prod_{j=1}^{i-1}\exp(-\delta_j \sigma_j)\right]  \left[1 - \exp(-\delta_i \sigma_i)\right] \mathbf{c}_i(\mathbf{d})

Where :math:`\delta_i, \sigma_i, \mathbf{c}_i` are segment i's length, density, and color, respectively. :math:`\mathbf{d}` is the viewing direction
and :math:`\mathbf{C}` is the final output color.

Also you can render rays directly, by using the forward method of VolumeRenderer:

>>> ray = svox.Rays(origins = ... dirs=..., viewdirs=...)
>>> ren(ray)

You can pass :code:`fast=True` to either render_persp or this forward method
to allow fast rendering (with early stopping) potentially at the cost of quality.

These functions are backed by CUDA analytic derivatives.
For example,

>>> im = ren.render_persp(camera)
>>> torch.abs(im - im_gt).mean().backward()
>>> print(tree.data.grad.shape)

Finally, NDC views are also internally supported in render_persp.
To use this features, pass :code:`ndc=svox.NDCConfig(width=..., heigh=..., focal=...)`
to the VolumeRenderer constructor.

**Troubleshooting**: If you get an error about a tensor being non-contiguous,
please make sure it is contiguous using
:code:`.contiguous()`, for example
:code:`svox.Rays(origins=r[:, :3].contiguous(), dirs=r[:, 3:6].contiguous(), viewdirs=r[:, 3:6].contiguous())`.

.. _leaf_level_acc:

Advanced Leaf-level Accessors
------------------------------
Some more functions for working with leaves

>>> tree.lengths  # Side lengths of each leaf voxel (same order as leaf matrix)
>>> tree.depths   # Depth of each leaf voxel (root is at **-1**)
>>> tree.corners  # Lowest corner of each leaf voxel
>>> tree.values   # Values at each leaf voxel
>>> tree.sample(n_samples: int)   # Sample uniformly random points in each voxel

In each case you may also use N3TreeView, for example

>>> tree[tree.depths==2].corners

For each of lengths/corners/sample there is also a \*_local version
which returns points and lengths in local coordinates :math:`[0,1]^3`.

Advanced: Volume Rendering Weight Accumulator Context
-------------------------------------------------------
Sometimes we want to accumulate volume rendering weights in each tree leaf,
to see how much each leaf voxel was used in the rendering process.
We may either want the max or total rendering weight (influence) within each voxel.
We have a built-in context manager to do so.

>>> with tree.accumulate_weights(op="sum") as accum:  # or op="max"
>>>     # Do some ray/image rendering with a renderer on the tree
>>>     # Tree cannot be refined or shrank here
>>> accum = accum()

The final :code:`accum` is a float tensor of shape
equal to tree.n_leaves which can
be used to index into the tree.
Each entry is equal to the *sum* of all volume rendering *weights*
for all rays which every hit the voxel within the context.
You can use it as follows:

>>> tree[accum > 1.0].refine()
>>> tree[accum < 1.0] += 1

*Advanced*: You can also use :code:`accum.value` to grab the complete
accumulated tensor of size equal to :code:`tree.data`.
This is more efficient than using :code:`accum()`.
