#  Copyright 2021 PlenOctree Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
Sparse voxel N^3 tree
"""

import os.path as osp
import torch
import numpy as np
from torch import nn, autograd
from svox.helpers import N3TreeView, DataFormat, _get_c_extension
from warnings import warn

_C = _get_c_extension()

class _QueryVerticalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree_spec, indices):
        out, node_ids = _C.query_vertical(tree_spec, indices)

        ctx.mark_non_differentiable(node_ids)
        ctx.tree_spec = tree_spec
        ctx.save_for_backward(indices)
        return out, node_ids

    @staticmethod
    def backward(ctx, grad_out, dummy):
        if ctx.needs_input_grad[0]:
            return _C.query_vertical_backward(ctx.tree_spec,
                         ctx.saved_tensors[0],
                         grad_out.contiguous()), None, None
        return None, None, None


class N3Tree(nn.Module):
    """
    PyTorch :math:`N^3`-tree library with CUDA acceleration.
    By :math:`N^3`-tree we mean a 3D tree with branching factor N at each interior node,
    where :math:`N=2` is the familiar octree.

    .. warning::
        `nn.Parameters` can change size, which
        makes current optimizers invalid. If any :code:`refine(): or
        :code:`shrink_to_fit()` call returns True,
        or :code:`expand(), shrink()` is used,
        please re-make any optimizers
    """
    def __init__(self, N=2, data_dim=None, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="RGBA",
            extra_data=None,
            device="cpu",
            dtype=torch.float32,
            map_location=None):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf (NEW in 0.2.28: optional if data_format other than RGBA is given).
                        If data_format = "RGBA" or empty, this defaults to 4.
        :param depth_limit: int maximum depth  of tree to stop branching/refining
                            Note that the root is at depth -1.
                            Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                            are depth 9. :code:`max_depth` applies to the same
                            depth values.
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
                            inital resolution will be :code:`[N^(init_refine + 1)]^3`.
                            initial max_depth will be init_refine.
        :param geom_resize_fact: float geometric resizing factor
        :param radius: float or list, 1/2 side length of cube (possibly in each dim)
        :param center: list center of space
        :param data_format: a string to indicate the data format. :code:`RGBA | SH# | SG# | ASG#`
        :param extra_data: extra data to include with tree
        :param device: str device to put data
        :param dtype: str tree data type, torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device (will override device and warn)

        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N : int = N

        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'

        self.data_format = DataFormat(data_format) if data_format is not None else None
        self.data_dim : int = data_dim
        self._maybe_auto_data_dim()
        del data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.register_parameter("data", nn.Parameter(
            torch.zeros(init_reserve, N, N, N, self.data_dim, dtype=dtype, device=device)))
        self.register_buffer("child", torch.zeros(
            init_reserve, N, N, N, dtype=torch.int32, device=device))
        self.register_buffer("parent_depth", torch.zeros(
            init_reserve, 2, dtype=torch.int32, device=device))

        self.register_buffer("_n_internal", torch.tensor(1, device=device))
        self.register_buffer("_n_free", torch.tensor(0, device=device))

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        radius = torch.tensor(radius, dtype=dtype, device=device)
        center = torch.tensor(center, dtype=dtype, device=device)

        self.register_buffer("invradius", 0.5 / radius)
        self.register_buffer("offset", 0.5 * (1.0 - center / radius))

        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact

        if extra_data is not None:
            assert isinstance(extra_data, torch.Tensor)
            self.register_buffer("extra_data", extra_data.to(dtype=dtype, device=device))
        else:
            self.extra_data = None

        self._ver = 0
        self._invalidate()
        self._lock_tree_structure = False
        self._weight_accum = None
        self._weight_accum_op = None

        self.refine(repeats=init_refine)


    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,

        :param indices: torch.Tensor :code:`(Q, 3)`
        :param values: torch.Tensor :code:`(Q, K)`
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.

        """
        assert len(indices.shape) == 2
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert not values.requires_grad  # Grad wrt values not supported
        indices = indices.to(device=self.data.device)
        values = values.to(device=self.data.device)

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow assignment")
            indices = self.world2tree(indices)

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                ind_floor.clamp_max_(self.N - 1)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self._spec(), indices, values)

    def forward(self, indices, cuda=True, want_node_ids=False, world=True):
        """
        Get tree values. Differentiable.

        :param indices: :code:`(Q, 3)` the points
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param want_node_ids: if true, returns node ID for each query.
        :param world: use world space instead of :code:`[0,1]^3`, default True

        :return: :code:`(Q, data_dim), [(Q)]`

        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert len(indices.shape) == 2

        if not cuda or _C is None or not self.data.is_cuda:
            if not want_node_ids:
                warn("Using slow query")
            if world:
                indices = self.world2tree(indices)

            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.empty((n_queries, self.data_dim), dtype=self.data.dtype,
                                  device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            ind = indices.clone()

            if want_node_ids:
                subidx = torch.zeros((n_queries, 3), dtype=torch.long, device=indices.device)

            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor

                sel = (node_ids[remain_indices], *(ind_floor.long().T),)

                deltas = self.child[sel]

                term_mask = deltas == 0
                term_indices = remain_indices[term_mask]

                vals = self.data[sel]
                result[term_indices] = vals[term_mask]
                if want_node_ids:
                    subidx[term_indices] = ind_floor.to(torch.long)[term_mask]

                node_ids[remain_indices] += deltas
                remain_indices = remain_indices[~term_mask]
                ind = ind[~term_mask]

            if want_node_ids:
                txyz = torch.cat([node_ids[:, None], subidx], axis=-1)
                return result, self._pack_index(txyz)

            return result
        else:
            result, node_ids = _QueryVerticalFunction.apply(
                                self.data, self._spec(world), indices);
            return (result, node_ids) if want_node_ids else result

    # Special features
    def snap(self, indices):
        """
        Snap indices to lowest corner of corresponding leaf voxel

        :param indices: :code:`(B, 3)` indices to snap

        :return: :code:`(B, 3)`

        """
        return self[indices].corners

    def partial(self, data_sel=None, data_format=None, dtype=None, device=None):
        """
        Get partial tree with some of the data dimensions (channels)
        E.g. :code:`tree.partial(-1)` to get tree with data_dim 1 of last channel only

        :param data_sel: data channel selector, default is all channels
        :param data_format: data format for new tree, default is current format
        :param dtype: new data type, torch.float32 | torch.float64
        :param device: where to put result tree

        :return: partial N3Tree (copy)
        """
        if device is None:
            device = self.data.device
        if data_sel is None:
            new_data_dim = self.data_dim
            sel_indices = None
        else:
            sel_indices = torch.arange(self.data_dim)[data_sel]
            if sel_indices.ndim == 0:
                sel_indices = sel_indices.unsqueeze(0)
            new_data_dim = sel_indices.numel()
        if dtype is None:
            dtype = self.data.dtype
        t2 = N3Tree(N=self.N, data_dim=new_data_dim,
                data_format=data_format or str(self.data_format),
                depth_limit=self.depth_limit,
                geom_resize_fact=self.geom_resize_fact,
                dtype=dtype,
                device=device)
        def copy_to_device(x):
            return torch.empty(x.shape, dtype=x.dtype, device=device).copy_(x)
        t2.invradius = copy_to_device(self.invradius)
        t2.offset = copy_to_device(self.offset)
        t2.child = copy_to_device(self.child)
        t2.parent_depth = copy_to_device(self.parent_depth)
        t2._n_internal = copy_to_device(self._n_internal)
        t2._n_free = copy_to_device(self._n_free)
        if self.extra_data is not None:
            t2.extra_data = copy_to_device(self.extra_data)
        else:
            t2.extra_data = None
        t2.data_format = self.data_format
        if data_sel is None:
            t2.data = nn.Parameter(copy_to_device(self.data.data))
        else:
            t2.data = nn.Parameter(copy_to_device(self.data.data[..., sel_indices].contiguous()))
        return t2

    def expand(self, data_format, data_dim=None, remap=None):
        """
        Modify the size of the data stored at the octree leaves.

        :param data_format: new data format, :code:`RGBA | SH# | SG# | ASG#`
        :param data_dim: data dimension; inferred from data_format by default
                only needed if data_format is RGBA.
        :param remap: mapping of old data to new data. For each leaf, we will do
                :code:`new_data[remap] = old_data` if the new data_dim
                is equal or larger, or `new_data = old_data[remap]` else. By default,
                this will be inferred automatically (maps basis functions
                in the correct way).

        .. warning::
                Will change the nn.Parameter size (data), breaking optimizer! Please re-create the optimizer
        """
        assert isinstance(data_format, str), "Please specify valid data format"
        old_data_format = self.data_format
        old_data_dim = self.data_dim
        self.data_format = DataFormat(data_format) if data_format is not None else None
        self.data_dim = data_dim
        self._maybe_auto_data_dim()
        del data_dim

        shrinking = self.data_dim < old_data_dim

        if remap is None:
            if shrinking:
                sigma_arr = torch.tensor([old_data_dim - 1])
                if old_data_format is None or self.data_format.format == DataFormat.RGBA:
                    remap = torch.cat([torch.arange(self.data_dim - 1), sigma_arr])
                else:
                    assert self.data_format.basis_dim >= 1, \
                           "Please manually specify data_dim for expand()"
                    old_basis_dim = old_data_format.basis_dim
                    if old_basis_dim < 0:
                        old_basis_dim = 1
                    shift = old_basis_dim
                    arr = torch.arange(self.data_format.basis_dim)
                    remap = torch.cat([arr, shift + arr, 2 * shift + arr, sigma_arr])
            else:
                sigma_arr = torch.tensor([self.data_dim-1])
                if old_data_format is None or self.data_format.format == DataFormat.RGBA:
                    remap = torch.cat([torch.arange(old_data_dim - 1), sigma_arr])
                else:
                    assert self.data_format.basis_dim >= 1, \
                           "Please manually specify data_dim for expand()"
                    old_basis_dim = old_data_format.basis_dim
                    if old_basis_dim < 0:
                        old_basis_dim = 1
                    shift = self.data_format.basis_dim
                    arr = torch.arange(old_basis_dim)
                    remap = torch.cat([arr, shift + arr, 2 * shift + arr, sigma_arr])

        may_oom = self.data.numel() > 8e9
        if may_oom:
            # Potential OOM prevention hack
            self.data = nn.Parameter(self.data.cpu())
        tmp_data = torch.zeros(
            (*self.data.data.shape[:-1], self.data_dim),
            dtype=self.data.dtype, device=self.data.device)
        if shrinking:
            tmp_data[:] = self.data.data[..., remap]
        else:
            tmp_data[..., remap] = self.data.data
        if may_oom:
            self.data = nn.Parameter(tmp_data.to(device=self.child.device))
        else:
            self.data = nn.Parameter(tmp_data)
        self._invalidate()

    def shrink(self, data_format, data_dim=None, remap=None):
        """
        Modify the size of the data stored at the octree leaves.
        (Alias of expand, because it can actually both expand and shrink)

        :param data_format: new data format, :code:`RGBA | SH# | SG# | ASG#`
        :param data_dim: data dimension; inferred from data_format by default
                only needed if data_format is RGBA.
        :param remap: mapping of old data to new data. For each leaf, we will do
                :code:`new_data = old_data[remap]` if the
                new data_dim gets smaller,
                or :code:`new_data[remap] = old_data` else. By default,
                this will be inferred automatically (maps basis functions
                in the correct way).

        .. warning::
                Will change the nn.Parameter size (data), breaking optimizer! Please re-create the optimizer
        """
        self.expand(data_format, data_dim, remap)

    def clone(self, device=None):
        """
        Deep copy the tree

        :param device: device of output tree (could e.g. copy cuda tree to cpu)

        """
        return self.partial(device=device)

    # 'Frontier' operations (node merging/pruning)
    def merge(self, frontier_sel=None, op=torch.mean):
        """
        Merge leaves into selected 'frontier' nodes
        (i.e., nodes for which all children are leaves).
        Use :code:`shrink_to_fit()` to recover memory freed.

        :param frontier_sel: selector (int, mask, list of indices etc)
                             for frontier nodes. In same order as :code:`reduce_frontier()`.
                             Default all nodes.
                             *Typical use*: use :code:`reduce_frontier(...)` to determine
                             conditions for merge, then pass
                             bool mask (of length :code:`n_frontier`) or indices to :code:`merge()`.
        :param op: reduction to combine child leaves into node.
                   E.g. torch.max, torch.mean.
                   Should take a positional argument :code:`x` :code:`(B, N, data_dim)` and
                   a named parameter :code:`dim` (always 1),
                   and return a matrix of :code:`(B, data_dim)`.
                   If a tuple is returned, uses first result.

        """
        if self.n_internal - self._n_free.item() <= 1:
            raise RuntimeError("Cannot merge root node")
        nid = self._frontier if frontier_sel is None else self._frontier[frontier_sel]
        if nid.numel() == 0:
            return False
        if nid.ndim == 0:
            nid = nid.reshape(1)
        data = self.data.data[nid]
        reduced_vals = op(data.view(-1, self.N ** 3, self.data_dim), dim=1)
        if isinstance(reduced_vals, tuple):
            # Allow torch.max, torch.min, etc
            reduced_vals = reduced_vals[0]
        parent_sel = (*self._unpack_index(self.parent_depth[nid, 0]).long().T,)
        self.data.data[parent_sel] = reduced_vals
        self.child[parent_sel] = 0
        self.parent_depth[nid] = -1
        self.child[nid] = -1
        self._n_free += nid.shape[0]
        self._invalidate()
        return True

    @property
    def n_frontier(self):
        """
        Number of frontier nodes
        """
        return self._frontier.shape[0]

    @property
    def frontier_depth(self):
        """
        Depth of frontier nodes, size :code:`(n_frontier)`
        """
        return self.parent_depth[self._frontier, 1]

    def reduce_frontier(self, op=torch.mean, dim=None, grad=False):
        """
        Reduce child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).

        :param op: reduction to combine child leaves into node.
                   E.g. torch.max, torch.mean.
                   Should take a positional argument :code:`x`
                   :code:`(B, N, in_dim <= data_dim)` and
                   a named parameter :code:`dim` (always 1),
                   and return a matrix of :code:`(B, your_out_dim)`.
        :param dim: dimension(s) of data to return, e.g. :code:`-1` returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :Example:

        .. code-block:: python

            def mean_and_var_func(x, dim=1):
                # (n_frontier, tree.N^3, data_dim) -> (n_frontier, 2 * data_dim)
                # Outputs mean and variance over children of each frontier node
                return torch.cat([torch.mean(x, dim=1),
                                  torch.var(x, dim=1)], dim=-1)
            # returns: (n_frontier, 2 * data_dim)
            tree.reduce_frontier(mean_and_var_func)

        :return: reduced tensor
        """
        nid = self._frontier
        if grad:
            data = self.data[nid]
        else:
            data = self.data.data[nid]
        data = data.view(-1, self.N ** 3, self.data_dim)
        if dim is None:
            return op(data, dim=1)
        else:
            return op(data[..., dim], dim=1)

    def max_frontier(self, dim=None, grad=False):
        """
        Takes max over child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).
        This is simply reduce_frontier with torch.max
        operation, taking the returned values and discarding the
        argmax part.

        :param dim: dimension(s) of data to return, e.g. :code:`-1` returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :return: reduced tensor
        """
        return self.reduce_frontier(op=lambda x, dim: torch.max(x, dim=dim)[0],
                grad=grad, dim=dim)

    def diam_frontier(self, dim=None, grad=False, scale=1.0):
        """
        Takes diameter over child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).

        :param dim: dimension(s) of data to return, e.g. :code:`-1` returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :return: reduced tensor
        """
        def diam_func(x, dim):
            # (B, N3, in_dim)
            if x.ndim == 2:
                x = x[:, :, None]
            N3 = x.shape[1]
            diam = torch.zeros(x.shape[:-2], device=x.device)
            for offset in range(N3):
                end_idx = -offset if offset > 0 else N3
                delta = (x[:, offset:] - x[:, :end_idx]) * scale
                n1 = torch.norm(delta, dim=-1).max(dim=-1)[0]
                if offset:
                    delta = (x[:, :offset] - x[:, end_idx:]) * scale
                    n2 = torch.norm(delta, dim=-1).max(dim=-1)[0]
                    n1 = torch.max(n1, n2)
                diam = torch.max(diam, n1)
            return diam

        return self.reduce_frontier(op=diam_func,
                grad=grad, dim=dim)


    def check_integrity(self):
        """
        Do some checks to verify the tree's structural integrity,
        mostly for debugging. Errors with message if check fails;
        does nothing else.
        """
        n_int = self.n_internal
        n_free = self._n_free.item()
        assert n_int - n_free > 0, "Tree has no root"
        assert self.data.shape[0] == self.capacity, "Data capacity mismatch"
        assert self.child.shape[0] == self.capacity, "Child capacity mismatch"
        assert (self.parent_depth[0] == 0).all(), "Node at index 0 must be root"

        free = self.parent_depth[:n_int, 0] == -1
        remain_ids = torch.arange(n_int, dtype=torch.long, device=self.child.device)[~free]
        remain_child = self.child[remain_ids]
        assert (remain_child >= 0).all(), "Nodes not topologically sorted"
        link_next = remain_child + remain_ids[..., None, None, None]

        assert link_next.max() < n_int, "Tree has an out-of-bounds child link"
        assert (self.parent_depth[link_next.reshape(-1), 0] != -1).all(), \
                "Tree has a child link to a deleted node"

        remain_ids = remain_ids[remain_ids != 0]
        if remain_ids.numel() == 0:
            return True
        remain_parents = (*self._unpack_index(
            self.parent_depth[remain_ids, 0]).long().T,)
        assert remain_parents[0].max() < n_int, "Parent link out-of-bounds (>=n_int)"
        assert remain_parents[0].min() >= 0, "Parent link out-of-bounds (<0)"
        for i in range(1, 4):
            assert remain_parents[i].max() < self.N, "Parent sublink out-of-bounds (>=N)"
            assert remain_parents[i].min() >= 0, "Parent sublink out-of-bounds (<0)"
        assert (remain_parents[0] + self.child[remain_parents] == remain_ids).all(), \
                "parent->child cycle consistency failed"
        return True

    @property
    def _frontier(self):
        """
        Get the nodes immediately above leaves (internal use)

        :return: node indices (first dim of self.data)
        """
        if self._last_frontier is None:
            node_selector = (self.child[ :self.n_internal] == 0).reshape(
                    self.n_internal, -1).all(dim=1)
            node_selector &= self.parent_depth[:self.n_internal, 0] != -1
            self._last_frontier = node_selector.nonzero(as_tuple=False).reshape(-1)
        return self._last_frontier


    # Leaf refinement & memory management methods
    def refine(self, repeats=1, sel=None):
        """
        Refine each selected leaf node, respecting depth_limit.

        :param repeats: int number of times to repeat refinement
        :param sel: :code:`(N, 4)` node selector. Default selects all leaves.

        :return: True iff N3Tree.data parameter was resized, requiring
                 optimizer reinitialization if you're using an optimizer

        .. warning::
            The parameter :code:`tree.data` can change due to refinement. If any refine() call returns True, please re-make any optimizers
            using :code:`tree.params()`.

        .. warning::
            The selector :code:`sel` is assumed to contain unique leaf indices. If there are duplicates
            memory will be wasted. We do not dedup here for efficiency reasons.

        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        with torch.no_grad():
            resized = False
            for repeat_id in range(repeats):
                filled = self.n_internal
                if sel is None:
                    # Default all leaves
                    sel = (*self._all_leaves().T,)
                depths = self.parent_depth[sel[0], 1]
                # Filter by depth & leaves
                good_mask = (depths < self.depth_limit) & (self.child[sel] == 0)
                sel = [t[good_mask] for t in sel]
                leaf_node =  torch.stack(sel, dim=-1).to(device=self.data.device)
                num_nc = len(sel[0])
                if num_nc == 0:
                    # Nothing to do
                    return False
                new_filled = filled + num_nc

                cap_needed = new_filled - self.capacity
                if cap_needed > 0:
                    self._resize_add_cap(cap_needed)
                    resized = True

                new_idxs = torch.arange(filled, filled + num_nc,
                        device=leaf_node.device, dtype=self.child.dtype) # NNC

                self.child[filled:new_filled] = 0
                self.child[sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
                self.data.data[filled:new_filled] = self.data.data[
                        sel][:, None, None, None]
                self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
                self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                        leaf_node[:, 0], 1] + 1  # depth

                if repeat_id < repeats - 1:
                    # Infer new selector
                    t1 = torch.arange(filled, new_filled,
                            device=self.data.device).repeat_interleave(self.N ** 3)
                    rangen = torch.arange(self.N, device=self.data.device)
                    t2 = rangen.repeat_interleave(self.N ** 2).repeat(
                            new_filled - filled)
                    t3 = rangen.repeat_interleave(self.N).repeat(
                            (new_filled - filled) * self.N)
                    t4 = rangen.repeat((new_filled - filled) * self.N ** 2)
                    sel = (t1, t2, t3, t4)
                self._n_internal += num_nc
        if repeats > 0:
            self._invalidate()
        return resized

    def _refine_at(self, intnode_idx, xyzi):
        """
        Advanced: refine specific leaf node. Mostly for testing purposes.

        :param intnode_idx: index of internal node for identifying leaf
        :param xyzi: tuple of size 3 with each element in :code:`{0, ... N-1}`
                    in xyz orde rto identify leaf within internal node

        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        assert min(xyzi) >= 0 and max(xyzi) < self.N
        if self.parent_depth[intnode_idx, 1] >= self.depth_limit:
            return

        xi, yi, zi = xyzi
        if self.child[intnode_idx, xi, yi, zi] != 0:
            # Already has child
            return

        resized = False
        filled = self.n_internal
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
            [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
        self.parent_depth[filled, 1] = depth
        self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi]
        self.data.data[intnode_idx, xi, yi, zi] = 0
        self._n_internal += 1
        self._invalidate()
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data,
        possibly dealing with fragmentation caused by merging.
        This is called by the :code:`save()` function by default, unless
        :code:`shrink=False` is specified there.

        .. warning::
                Will change the nn.Parameter size (data), breaking optimizer!
        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        n_int = self.n_internal
        n_free = self._n_free.item()
        new_cap = n_int - n_free
        if new_cap >= self.capacity:
            return False
        if n_free > 0:
            # Defragment
            free = self.parent_depth[:n_int, 0] == -1
            csum = torch.cumsum(free, dim=0)

            remain_ids = torch.arange(n_int, dtype=torch.long)[~free]
            remain_parents = (*self._unpack_index(
                self.parent_depth[remain_ids, 0]).long().T,)

            # Shift data over
            par_shift = csum[remain_parents[0]]
            self.child[remain_parents] -= csum[remain_ids] - par_shift
            self.parent_depth[remain_ids, 0] -= par_shift * (self.N ** 3)

            # Remake the data now
            self.data = nn.Parameter(self.data.data[remain_ids])
            self.child = self.child[remain_ids]
            self.parent_depth = self.parent_depth[remain_ids]
            self._n_internal.fill_(new_cap)
            self._n_free.zero_()
        else:
            # Direct resize
            self.data = nn.Parameter(self.data.data[:new_cap])
            self.child = self.child[:new_cap]
            self.parent_depth = self.parent_depth[:new_cap]
        self._invalidate()
        return True

    # Misc
    @property
    def n_leaves(self):
        return self._all_leaves().shape[0]

    @property
    def n_internal(self):
        return self._n_internal.item()

    @property
    def capacity(self):
        return self.parent_depth.shape[0]

    @property
    def max_depth(self):
        """
        Maximum tree depth - 1
        """
        return torch.max(self.depths).item()

    def accumulate_weights(self, op : str='sum'):
        """
        Begin weight accumulation.

        :param op: reduction to apply weight in each voxel,
                   sum | max

        .. warning::

            Weight accumulator has not been validated
            and may have bugs

        :Example:

        .. code-block:: python

            with tree.accumulate_weights() as accum:
                ...

            # (n_leaves) in same order as values etc.
            accum = accum()
        """
        return WeightAccumulator(self, op)

    # Persistence
    def save(self, path, shrink=True, compress=True):
        """
        Save to from npz file

        :param path: npz path
        :param shrink: if True (default), applies shrink_to_fit before saving
        :param compress: whether to compress the npz; may be slow

        """
        if shrink:
            self.shrink_to_fit()
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child.cpu(),
            "parent_depth" : self.parent_depth.cpu(),
            "n_internal" : self._n_internal.cpu().item(),
            "n_free" : self._n_free.cpu().item(),
            "invradius3" : self.invradius.cpu(),
            "offset" : self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "data": self.data.data.half().cpu().numpy()  # save CPU Memory
        }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def load(cls, path, device='cpu', dtype=torch.float32, map_location=None):
        """
        Load from npz file

        :param path: npz path
        :param device: str device to put data
        :param dtype: str torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device

        """
        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        tree = cls(dtype=dtype, device=device)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.child = torch.from_numpy(z["child"]).to(device)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        tree._n_internal.fill_(z["n_internal"].item())
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                                np.float32)).to(device)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(device)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(device)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        tree.data_format = DataFormat(z['data_format'].item()) if \
                'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
                          'extra_data' in z.files else None
        return tree

    # Magic
    def __repr__(self):
        return (f"svox.N3Tree(N={self.N}, data_dim={self.data_dim}, " +
                f"depth_limit={self.depth_limit}, " +
                f"capacity:{self.n_internal - self._n_free.item()}/{self.capacity}, " +
                f"data_format:{self.data_format or 'RGBA'})");

    def __getitem__(self, key):
        """
        Get N3TreeView
        """
        return N3TreeView(self, key)

    def __setitem__(self, key, val):
        N3TreeView(self, key).set(val)

    def __iadd__(self, val):
        self[:] += val
        return self

    def __isub__(self, val):
        self[:] -= val
        return self

    def __imul__(self, val):
        self[:] *= val
        return self

    def __idiv__(self, val):
        self[:] /= val
        return self

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return torch.Size((self.n_leaves, self.data_dim))

    def size(self, dim):
        return self.data_dim if dim == 1 else self.n_leaves

    def numel(self):
        return self.data_dim * self.n_leaves

    def __len__(self):
        return self.n_leaves

    # Internal utils
    def _calc_corners(self, nodes, cuda=True):
        if _C is not None and cuda and self.data.is_cuda:
            return _C.calc_corners(self._spec(), nodes.to(self.data.device))

        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3, device=curr.device, dtype=self.data.dtype)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask.clone()] = good_mask

            curr = self._unpack_index(self.parent_depth[curr[good_mask, 0], 0].long())

        return output


    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat //= self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        may_oom = self.capacity + cap_needed > 1e7  # My CPU Memory is limited
        if may_oom:
            # Potential OOM prevention hack
            self.data = nn.Parameter(self.data.cpu())
        self.data = nn.Parameter(torch.cat((self.data.data,
                        torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                dtype=self.data.dtype,
                                device=self.data.device)), dim=0))
        if may_oom:
            self.data = nn.Parameter(self.data.to(device=self.child.device))
        self.child = torch.cat((self.child,
                                torch.zeros((cap_needed, *self.child.shape[1:]),
                                   dtype=self.child.dtype,
                                   device=self.data.device)))
        self.parent_depth = torch.cat((self.parent_depth,
                                torch.zeros((cap_needed, *self.parent_depth.shape[1:]),
                                   dtype=self.parent_depth.dtype,
                                   device=self.data.device)))

    def _make_val_tensor(self, val):
        val_tensor = torch.tensor(val, dtype=self.data.dtype,
            device=self.data.device)
        while len(val_tensor.shape) < 2:
            val_tensor = val_tensor[None]
        if val_tensor.shape[-1] == 1:
            val_tensor = val_tensor.expand(-1, self.data_dim).contiguous()
        else:
            assert val_tensor.shape[-1] == self.data_dim
        return val_tensor

    def _all_leaves(self):
        if self._last_all_leaves is None:
            self._last_all_leaves = (self.child[
                :self.n_internal] == 0).nonzero(as_tuple=False).cpu()
        return self._last_all_leaves

    def world2tree(self, indices):
        """
        Scale world points to tree (:math:`[0,1]^3`)
        """
        return torch.addcmul(self.offset, indices, self.invradius)

    def tree2world(self, indices):
        """
        Scale tree points (:math:`[0,1]^3`) to world accoording to center/radius
        """
        return (indices  - self.offset) / self.invradius

    def _invalidate(self):
        self._ver += 1
        self._last_all_leaves = None
        self._last_frontier = None

    def _spec(self, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
        tree_spec.data = self.data
        tree_spec.child = self.child
        tree_spec.parent_depth = self.parent_depth
        tree_spec.extra_data = self.extra_data if self.extra_data is not None else \
                torch.empty((0, 0), dtype=self.data.dtype, device=self.data.device)
        tree_spec.offset = self.offset if world else torch.tensor(
                  [0.0, 0.0, 0.0], dtype=self.data.dtype, device=self.data.device)
        tree_spec.scaling = self.invradius if world else torch.tensor(
                  [1.0, 1.0, 1.0], dtype=self.data.dtype, device=self.data.device)
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                    self._weight_accum is not None else torch.empty(
                            0, dtype=self.data.dtype, device=self.data.device)
            tree_spec._weight_accum_max = (self._weight_accum_op == 'max')
        return tree_spec

    def _maybe_auto_data_dim(self):
        if self.data_format is not None and self.data_format.data_dim is not None:
            if self.data_dim is None:
                self.data_dim = self.data_format.data_dim
            else:
                assert self.data_format.data_dim == self.data_dim, "data_dim invalid for given data format"
        elif self.data_dim is None:
            # Legacy default
            self.data_dim = 4


# Redirect functions to N3TreeView so you can do tree.depths instead of tree[:].depths
def _redirect_to_n3view():
    redir_props = ['depths', 'lengths', 'lengths_local', 'corners', 'corners_local',
                   'values', 'values_local']
    redir_funcs = ['sample', 'sample_local', 'aux',
            'normal_', 'clamp_', 'uniform_', 'relu_', 'sigmoid_', 'nan_to_num_']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_func)(*args, **kwargs)
        setattr(N3Tree, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    def redirect_prop(redir_prop):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_prop)
        setattr(N3Tree, redir_prop, property(redir_impl))
    for redir_prop in redir_props:
        redirect_prop(redir_prop)
_redirect_to_n3view()

class WeightAccumulator():
    def __init__(self, tree, op):
        assert op in ['sum', 'max'], 'Unsupported accumulation'
        self.tree = tree
        self.op = op

    def __enter__(self):
        self.tree._lock_tree_structure = True
        self.tree._weight_accum = torch.zeros(
                self.tree.child.shape, dtype=self.data.dtype,
                device=self.tree.data.device)
        self.tree._weight_accum_op = self.op
        self.weight_accum = self.tree._weight_accum
        return self

    def __exit__(self, type, value, traceback):
        self.tree._weight_accum = None
        self.tree._weight_accum_op = None
        self.tree._lock_tree_structure = False

    @property
    def value(self):
        return self.weight_accum

    def __call__(self):
        return self.tree.aux(self.weight_accum)
