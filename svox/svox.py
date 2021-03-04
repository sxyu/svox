#  [BSD 2-CLAUSE LICENSE]
#
#  Copyright SVOX Authors 2021
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
        out, data_ids, node_ids = _C.query_vertical(tree_spec, indices)

        ctx.mark_non_differentiable(data_ids)
        ctx.mark_non_differentiable(node_ids)
        ctx.tree_spec = tree_spec
        ctx.save_for_backward(indices)
        return out, data_ids, node_ids

    @staticmethod
    def backward(ctx, grad_out, dummy, dummy2):
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
    makes current optimizers invalid. If data size changes,
    please re-make any optimizers. To handle this, you may pass
    a function taking no arguments through on_invalidate;
    this will be called each time data is resized.
    """
    def __init__(self, N=2, data_dim=4, depth_limit=10,
            init_refine=0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="RGBA",
            extra_data=None,
            map_location="cpu",
            on_invalidate=None):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf
        :param depth_limit: int maximum depth of tree to stop branching/refining
        :param init_refine: int number of times to refine entire tree initially
        :param radius: float or list, 1/2 side length of cube (possibly in each dim)
        :param center: list center of space
        :param data_format: a string to indicate the data format
        :param extra_data: extra data to include with tree
        :param map_location: str device to put data
        :param on_invalidate: optional callback when tree is invalidated
        (so that nn.Parameters should be remade)

        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N : int = N
        self.data_dim : int = data_dim
        N3 = N ** 3

        # Data buffer
        self.register_parameter("data", nn.Parameter(
            torch.zeros(1, data_dim, device=map_location)))
        # Maps indices in data buffer to node IDs
        #  also, -1 = free space, to be reused or recovered by shrink_to_fit()
        #  also, -2 = reserved
        self.register_buffer("back_link",
            torch.full((1,), fill_value=-2,
                dtype=torch.int32, device=map_location))


        # Maps nodes to next nodes (ofset) + maps leaves to data array (<=0 indices)
        self.register_buffer("child",
             torch.zeros(1, N, N, N, dtype=torch.int32, device=map_location))
        # Link to parent node
        self.register_buffer("parent", torch.zeros(
            1, dtype=torch.int32, device=map_location))
        # Depths in tree. 0 = child of root so it seems 1 off
        self.register_buffer("idepths", torch.zeros(
            1, dtype=torch.int32, device=map_location))

        # Translate & scale
        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        radius = torch.tensor(radius, dtype=torch.float32, device=map_location)
        center = torch.tensor(center, dtype=torch.float32, device=map_location)

        self.register_buffer("scaling", 0.5 / radius)
        self.register_buffer("offset", 0.5 * (1.0 - center / radius))

        self.depth_limit = depth_limit
        self.data_format = DataFormat(data_format or "")
        self.on_invalidate = on_invalidate

        if extra_data is not None:
            assert isinstance(extra_data, torch.Tensor)
            self.register_buffer("extra_data", extra_data.to(device=map_location))
        else:
            self.extra_data = None

        self._ver = 0
        self._lock_tree_data = False
        self._weight_accum = None

        self.refine(repeats=init_refine)


    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,

        :param indices: torch.Tensor (Q, 3)
        :param values: torch.Tensor (Q, K)
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.

.. warning::
        If lazy refinement was performed,
        `nn.Parameters` can change size due to set(),

.. warning::
        If multiple indices point to same leaf node,
        only one of them will be taken
        """
        assert len(indices.shape) == 2
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert not values.requires_grad  # Grad wrt values not supported
        assert indices.size(0) == values.size(0)
        indices = indices.to(device=self.data.device)
        values = values.to(device=self.data.device)

        self._maybe_lazy_alloc(indices)

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow assignment")
            indices = self.world2tree(indices)

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor

                sel = (node_ids, *(ind_floor.long().T),)

                deltas = self.child[sel]

                term_mask = deltas <= 0

                term_indices = remain_indices[term_mask]
                self.data.data[(-deltas[term_mask]).long()] = values[term_indices]

                nonterm_mask = ~term_mask
                node_ids = node_ids[nonterm_mask]
                node_ids += deltas[nonterm_mask]
                remain_indices = remain_indices[nonterm_mask]
                ind = ind[nonterm_mask]
        else:
            _C.assign_vertical(self._spec(), indices, values)

    def forward(self, indices, cuda=True, want_node_ids=False,
                want_data_ids=False, world=True):
        """
        Get tree values. Differentiable.

        :param indices: :math:`(Q, 3)` the points
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param want_node_ids: if true, returns node ID for each query. This is
                              _pack_index called on 4D index in child array.
        :param want_data_ids: if true, returns data storage index for each query.
                              This is leaf_id + 1, where leaf_id is
                              the index you would use in the key:
                              tree[key] to construct a N3TreeView.
        :param world: use world space instead of :math:`[0,1]^3`, default True

        :return: (Q, data_dim), [(Q)]

        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert indices.ndim == 2

        if not cuda or _C is None or not self.data.is_cuda:
            if not want_node_ids:
                warn("Using slow query")
            if world:
                indices = self.world2tree(indices)

            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.empty((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            ind = indices.clone()

            term_data_idx, term_node_idx = None, None
            if want_data_ids:
                term_data_idx = torch.empty((n_queries,), dtype=torch.int32,
                        device=indices.device)
            if want_node_ids:
                term_node_idx = torch.empty((n_queries,), dtype=torch.int32,
                        device=indices.device)

            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor
                ind_floor = ind_floor.long()

                sel = (node_ids, *ind_floor.unbind(-1),)

                deltas = self.child[sel]

                term_mask = deltas <= 0
                term_indices = remain_indices[term_mask]

                data_indices = -deltas[term_mask]
                result[term_indices] = self.data[data_indices.long()]
                if want_data_ids:
                    term_data_idx[term_indices] = data_indices
                if want_node_ids:
                    self_id = self._pack_index(torch.cat((
                                node_ids[term_mask, None],
                                ind_floor[term_mask]
                            ), dim=-1).int())
                    term_node_idx[term_indices] = self_id

                nonterm_mask = ~term_mask
                node_ids = node_ids[nonterm_mask]
                node_ids += deltas[nonterm_mask]
                remain_indices = remain_indices[nonterm_mask]
                ind = ind[nonterm_mask]

        else:
            result, term_data_idx, term_node_idx = _QueryVerticalFunction.apply(
                                self.data, self._spec(world), indices);

        ret = [result]
        if want_data_ids:
            ret.append(term_data_idx)
        if want_node_ids:
            ret.append(term_node_idx)
        if len(ret) == 1:
            return ret[0]
        return ret

    # Special features
    def partial(self, data_sel=None):
        """
        Get partial tree with some of the data dimensions (channels)
        E.g. tree.partial(-1) to get tree with data_dim 1 of last channel only
        :param data_sel: data channel selector, default is all channels
        :return: partial N3Tree (copy)
        """
        if data_sel is None:
            new_data_dim = self.data_dim
        else:
            sel_indices = torch.arange(self.data_dim)[data_sel]
            if sel_indices.ndim == 0:
                sel_indices = sel_indices.unsqueeze(0)
            new_data_dim = sel_indices.numel()
        t2 = N3Tree(N=self.N, data_dim=new_data_dim,
                depth_limit=self.depth_limit,
                geom_resize_fact=self.geom_resize_fact,
                map_location=self.data.data.device)
        t2.scaling = self.scaling.clone()
        t2.offset = self.offset.clone()
        t2.child = self.child.clone()
        t2.parent = self.parent.clone()
        t2.idepths = self.idepths.clone()
        t2.back_link = self.back_link.clone()
        if data_sel is None:
            t2.data.data = self.data.data.clone()
        else:
            t2.data.data = self.data.data[..., sel_indices].contiguous()
        return t2

    def clone(self):
        """
        Deep copy the tree
        """
        return self.partial()


    # Leaf refinement & memory management methods
    def refine(self, repeats=1, sel=None, lazy=False):
        """
        Refine each selected leaf node, respecting depth_limit.

        :param repeats: int number of times to repeat refinement
        :param sel: (N, 4) node selector. Default selects all leaves.
        :param lazy: if True, this will NOT actually modify data memory
                     instead that is done lazily when required (data is modified)
                     Note that when lazy refinement is used, memory may be saved,
                     but indexing by data indices is not supported.

        :return: True iff N3Tree.data parameter was resized, requiring
                 optimizer reinitialization if you're using an optimizer

.. warning::
        Unless lazy=True, `nn.Parameters` can change size

.. warning::
    The selector :code:`sel` is assumed to contain unique leaf indices. If there are duplicates
    memory will be wasted. We do not dedup here for efficiency reasons.
        """
        with torch.no_grad():
            resized = False
            if sel is None:
                # Default all leaves
                sel = (self.child <= 0).nonzero(as_tuple=True)

            for repeat_id in range(repeats):
                filled = self.n_internal
                depths = self.idepths[sel[0]]
                # Filter by depth & leaves
                good_mask = (depths < self.depth_limit) & (self.child[sel] <= 0)
                sel = [t[good_mask] for t in sel]
                leaf_node = torch.stack(sel, dim=-1)
                num_nc = len(sel[0])
                if num_nc == 0:
                    # Nothing to do
                    return resized
                new_filled = filled + num_nc

                self._resize_add_cap(num_nc)

                new_idxs = torch.arange(filled, new_filled,
                        device=self.child.device, dtype=self.child.dtype) # NNC

                data_idxs = -self.child[sel]
                ldata_idxs = data_idxs.long()

                # Old nodes child link to new nodes
                self.child[sel] = new_idxs - sel[0].to(torch.int32)
                # New nodes parent link to old nodes + position
                self.parent[filled:new_filled] = self._pack_index(leaf_node)
                # Increment depths
                self.idepths[filled:new_filled] = self.idepths[sel[0]] + 1
                # Mark data memory as free
                self.back_link[ldata_idxs] = -1
                self.back_link[0] = -2  # Never free 'null' element

                if lazy:
                    # New leaves have null memory
                    self.child[filled:new_filled] = 0
                else:
                    # Find free memory
                    free_indices = torch.where(self.back_link == -1)[0]

                    old_data = self.data.data[ldata_idxs].clone()

                    # Resize to correct size
                    dfilled = self.n_leaves + 1
                    N3 = self.N ** 3
                    dneeded = max(num_nc * N3 - free_indices.numel(), 0)
                    new_dfilled = dfilled + dneeded
                    self._resize_add_data_cap(dneeded)

                    free_indices = torch.cat((free_indices,
                                              torch.arange(dfilled, new_dfilled,
                                                  device=self.data.device, dtype=torch.long)))

                    # Assign values to new nodes equal to old (parent) nodes
                    self.data.data[free_indices] = torch.repeat_interleave(
                            old_data, N3, 0)
                    del old_data

                    # Update back links (which point from data vector to new nodes)
                    self.back_link[free_indices] = torch.arange(filled * N3,
                                                                new_filled * N3,
                                                                dtype=torch.int32,
                                                                device=self.data.device)

                    # Update tree leaf pointers on new nodes (to point to data vector)
                    # <= 0 means position on data vector
                    self.child[filled:new_filled] = -free_indices.view(
                            -1, self.N, self.N, self.N)
                    resized = True

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
        if resized:
            self._invalidate()
        return resized

    # Misc
    @property
    def n_internal(self):
        return self.child.size(0)

    @property
    def n_leaves(self):
        return self.data.size(0) - 1

    @property
    def capacity(self):
        return self.n_internal

    @property
    def max_depth(self):
        """
        Maximum tree depth - 1.
        Note: need to look thru leaves
        """
        return torch.max(self.idepths).item()

    @property
    def n_freeable(self):
        """
        Number of data entries which can be freed using shrink_to_fit
        Note: need to look thru leaves
        """
        return (self.back_link == -1).sum().item()

    def accumulate_weights(self):
        """
        Begin weight accumulation

.. code-block:: python

        with tree.accumulate_weights() as accum:
            ...

        # (n_leaves) in same order as values etc.
        accum = accum()
        """
        return WeightAccumulator(self)

    # Persistence
    def save(self, path, compress=False, strip=False):
        """
        Save to from npz file

        :param path: npz path
        :param compress: whether to compress the npz; may be slow
        :param strip: whether to only save attributes needed for rendering

        """
        data = {
            "child" : self.child.cpu(),
            "scaling" : self.scaling.cpu(),
            "offset" : self.offset.cpu(),
            "data": self.data.data.cpu().numpy().astype(np.float16)
        }
        if not strip:
            data.update({
                "parent" : self.parent.cpu(),
                "depths" : self.idepths.cpu(),
                "back_link": self.back_link.cpu(),
                "depth_limit": self.depth_limit,
            })
        data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def load(cls, path, map_location='cpu'):
        """
        Load from npz file

        :param path: npz path
        :param map_location: device to put data

        """
        tree = cls(map_location=map_location)
        z = np.load(path)

        tree.child = torch.from_numpy(z["child"]).to(map_location)
        tree.N = tree.child.shape[-1]
        tree.extra_data = torch.from_numpy(z['extra_data']).to(map_location) if \
                          'extra_data' in z.files else None
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(map_location)
        tree.depth_limit = int(z.get("depth_limit", 100))

        if "parent_depth" in z.files:
            warn("Converting legacy data buffer format (pre 0.3.0); " +
                 "this is deprecated and may be removed")
            if "invradius3" in z.files:
                tree.scaling = torch.from_numpy(z["invradius3"].astype(
                                    np.float32)).to(map_location)
            else:
                tree.scaling.fill_(z["invradius"].item())
            parent_depth = torch.from_numpy(z["parent_depth"]).to(map_location)
            tree.parent = parent_depth[:, 0]
            tree.idepths = parent_depth[:, 1]
            data_buf_sz = tree.child.numel()

            tree.back_link = torch.full((data_buf_sz,), fill_value=-1,
                    dtype=torch.int32,
                    device=map_location)
            leaves = (tree.child == 0).nonzero(as_tuple=False)
            leaf_indices = tree._pack_index(leaves)
            ileaf_indices = leaf_indices.int()
            tree.child[leaves.unbind(-1)] = -ileaf_indices
            tree.back_link[leaf_indices] = ileaf_indices
            tmp_data = torch.from_numpy(z["data"].astype(np.float32))
            tree.data = nn.Parameter(tmp_data.reshape(
                -1, tmp_data.shape[-1]).to(map_location))
            zero_data_mask = (tree.data == 0.0).all(dim=1) & (tree.back_link >= 0)
            tree.child[tree._unpack_index(tree.back_link[zero_data_mask]).long(
                ).unbind(-1)] = 0

            tree.back_link[zero_data_mask] = -1
            tree.back_link[0] = -2
        else:
            if 'parent' in z.files:
                tree.parent = torch.from_numpy(z["parent"]).to(map_location)
            else:
                warn("Stripped N3Tree, only querying/assignment/rendering supported")
            if 'depths' in z.files:
                tree.idepths = torch.from_numpy(z["depths"]).to(map_location)
            tree.scaling = torch.from_numpy(z["scaling"].astype(
                                np.float32)).to(map_location)
            tree.data = nn.Parameter(torch.from_numpy(
                z["data"].astype(np.float32)).to(map_location))
            if 'back_link' in z.files:
                tree.back_link = torch.from_numpy(z["back_link"]).to(map_location)
        tree.data_dim = tree.data.data.shape[-1]
        tree.data_format = DataFormat(z['data_format'].item()) if \
                'data_format' in z.files else None
        if tree.data_format is None:
            warn("Legacy N3Tree (pre 0.2.18) without data_format, auto-infering SH order")
            # Auto SH order
            ddim = tree.data_dim
            if ddim == 4:
                tree.data_format = DataFormat("")
            else:
                tree.data_format = DataFormat(f"SH{(ddim - 1) // 3}")
        return tree

    def shrink_to_fit(self):
        """
        Free all unused space.
        """
        free = self.back_link == -1
        csum = torch.cumsum(free, dim=0)
        if csum[-1] == 0:
            return False
        nonfree = ~free
        del free

        nonfree[0] = True
        packed = self.back_link[nonfree]
        leaf_sel = self._unpack_index(
             self.back_link[nonfree]).long().unbind(-1)

        self.child[leaf_sel] += csum[nonfree]
        del leaf_sel
        self.data = torch.nn.Parameter(self.data.data[nonfree])
        self.back_link = self.back_link[nonfree]
        self._invalidate()
        return True

    def quantize_median_cut(self, order):
        assert _C is not None  # Need C extension
        # Get rid of bogus elements
        self.shrink_to_fit()
        device = self.data.device
        # Currently implemented on CPU
        colors, color_map = _C.quantize_median_cut(self.data.data[1:].cpu(), order)

        child = self.child.cpu()
        leaf_sel = (child < 0).nonzero(as_tuple=True)
        child[leaf_sel] = -color_map[-child[leaf_sel].long() - 1] + 1

        self.data = nn.Parameter(colors.to(device=device))
        self.child = child.to(device=device)
        self._invalidate()

    # Magic
    def __repr__(self):
        return (f"svox.N3Tree(N={self.N}, data_dim={self.data_dim}, " +
                f"depth_limit={self.depth_limit}, " +
                f"capacity={self.capacity}, " +
                f"n_leaves={self.n_leaves}, " +
                f"data_format={self.data_format or 'RGBA'})");

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

    # Internal utils
    def _calc_corners(self, nodes):
        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3, device=curr.device, dtype=torch.float32)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask] = good_mask

            curr = self._unpack_index(self.parent[curr[good_mask, 0]].long())

        return output


    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat /= self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing node storage capacity
        """
        self.child = torch.cat((self.child,
                                torch.zeros((cap_needed, *self.child.shape[1:]),
                                   dtype=self.child.dtype,
                                   device=self.data.device)))
        self.idepths = torch.cat((self.idepths,
                                torch.zeros(cap_needed,
                                   dtype=self.idepths.dtype,
                                   device=self.data.device)))
        self.parent = torch.cat((self.parent,
                                torch.zeros(cap_needed,
                                   dtype=self.parent.dtype,
                                   device=self.data.device)))

    def _resize_add_data_cap(self, cap_needed):
        """
        Helper for increasing data storage capacity
        """
        self.data = nn.Parameter(torch.cat((self.data.data,
                        torch.zeros((cap_needed, self.data_dim),
                                dtype=self.data.dtype,
                                device=self.data.device)), dim=0))
        self.back_link = torch.cat((self.back_link,
                        torch.zeros(cap_needed,
                                dtype=self.back_link.dtype,
                                device=self.data.device)), dim=0)

    def _maybe_lazy_alloc(self, indices):
        """
        Lazily allocate memory for 'virtual' nodes
        which were created by refine()
        """
        with torch.no_grad():
            vals, data_ids, node_ids = self(indices, want_data_ids=True, want_node_ids=True)

            # Only consider unique nodes
            node_ids, inv = torch.unique(node_ids, return_inverse=True)
            data_ids_tmp = torch.empty(node_ids.size(0), dtype=data_ids.dtype,
                    device=indices.device)
            data_ids_tmp[inv] = data_ids
            data_ids = data_ids_tmp.long()
            del data_ids_tmp

            vals_tmp = torch.empty((node_ids.size(0), vals.size(1)), dtype=vals.dtype,
                    device=indices.device)
            vals_tmp[inv] = vals
            vals = vals_tmp
            del vals_tmp

            is_lazy = data_ids == 0

            vals = vals[is_lazy]
            if vals.numel() == 0:
                return data_ids
            if self._lock_tree_data:
                raise RuntimeError("Tree locked")
            node_ids = node_ids[is_lazy]

            # Find free memory
            free_indices = torch.where(self.back_link == -1)[0]

            dfilled = self.n_leaves + 1
            dneeded = max(vals.size(0) - free_indices.numel(), 0)
            new_dfilled = dfilled + dneeded
            self._resize_add_data_cap(dneeded)

            free_indices = torch.cat((free_indices,
                                      torch.arange(dfilled, new_dfilled,
                                          device=self.data.device, dtype=torch.long)))

            self.back_link[free_indices] = node_ids
            self.child[self._unpack_index(node_ids).long().unbind(-1)] = -free_indices.int()
            data_ids[is_lazy] = free_indices
            self.data[free_indices] = vals

            self._invalidate()

            data_ids = data_ids[inv]
            return data_ids
        return True


    def world2tree(self, indices):
        """
        Scale world points to tree (:math:`[0,1]^3`)
        """
        return torch.addcmul(self.offset, indices, self.scaling)

    def tree2world(self, indices):
        """
        Scale tree points (:math:`[0,1]^3`) to world accoording to center/radius
        """
        return (indices  - self.offset) / self.scaling

    def nan_to_num_(self, inf_val=2e4):
        """
        Convert nans to 0.0 and infs to inf_val
        """
        self.data.data[torch.isnan(self.data.data)] = 0.0
        inf_mask = torch.isinf(self.data.data)
        self.data.data[inf_mask & (self.data.data > 0)] = inf_val
        self.data.data[inf_mask & (self.data.data < 0)] = -inf_val

    def set_default(self, value):
        """
        Set default value for nodes with no data
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.data.device, dtype=torch.float32)
        self.data.data[0] = value.to(device=self.data.device)

    def _invalidate(self):
        self._ver += 1
        if self.on_invalidate is not None:
            self.on_invalidate()

    def _spec(self, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
        tree_spec.data = self.data
        tree_spec.child = self.child
        tree_spec.extra_data = self.extra_data if self.extra_data is not None else \
                torch.empty((0, 0), device=self.data.device)
        tree_spec.offset = self.offset if world else torch.tensor(
                  [0.0, 0.0, 0.0], device=self.data.device)
        tree_spec.scaling = self.scaling if world else torch.tensor(
                  [1.0, 1.0, 1.0], device=self.data.device)
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                    self._weight_accum is not None else torch.empty(
                            0, device=self.data.device)
        return tree_spec

    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat //= self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def size(self, dim):
        return self.data_dim if dim == 1 else self.n_leaves

    def dim(self):
        return 2

    def numel(self):
        return self.data_dim * self.n_leaves

    @property
    def shape(self):
        return torch.Size((self.n_leaves, self.data_dim))

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return self.n_leaves

# Redirect functions to N3TreeView so you can do tree.depths instead of tree[:].depths
def _redirect_to_n3view():
    redir_props = ['lengths', 'lengths_local', 'corners', 'corners_local',
                   'values', 'values_nograd', 'depths']
    redir_funcs = ['sample', 'sample_local', 'aux',
            'normal_', 'clamp_', 'uniform_', 'relu_', 'sigmoid_',
            'clamp_', 'clamp_min_', 'clamp_max_', 'sqrt_', 'normal_', 'uniform_']
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
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        self.tree._lock_tree_data = True
        self.tree._weight_accum = torch.zeros(
                self.tree.child.shape, dtype=torch.float32,
                device=self.tree.data.device)
        self.weight_accum = self.tree._weight_accum
        return self

    def __exit__(self, type, value, traceback):
        self.tree._weight_accum = None
        self.tree._lock_tree_data = False

    @property
    def value(self):
        return self.weight_accum

    def __call__(self):
        return self.tree.aux(self.weight_accum)
