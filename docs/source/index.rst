.. svox documentation master file, created by
   sphinx-quickstart on Wed Feb 17 23:40:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the svox documentation
===================================

This is a PlenOctree volume rendering implementation as a *PyTorch* extension with CUDA acceleration. The code is availble in `this repo <https://github.com/sxyu/svox/>`_.
This is part of the code release of

| PlenOctrees for Real Time Rendering of Neural Radiance Fields
| Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

| Paper: `https://arxiv.org/abs/2103.14024 <https://arxiv.org/abs/2103.14024>`_
| Website: `https://alexyu.net/plenoctrees <https://alexyu.net/plenoctrees>`_
| Source code (SVOX): `https://github.com/sxyu/svox <https://github.com/sxyu/svox>`_

Install with :code:`pip install svox`.

>>> import svox
>>> tree = svox.N3Tree(data_dim=4)
>>> print(tree)
svox.N3Tree(N=2, data_dim=4, depth_limit=10; capacity:1/1 max_depth:0)
>>> tree.to('cuda:0')


:code:`data_dim` is the size of data stored at each leaf.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick
   svox
   ex_nerf
