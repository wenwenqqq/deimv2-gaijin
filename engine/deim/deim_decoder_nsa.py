"""
DEIMv2 decoder variant whose query self-attention is replaced by NSA
(Native Sparse Attention, Yuan et al., 2025 - arXiv:2502.11089).

NSA replaces the dense key/value set of self-attention with a dynamically
constructed, information-dense set and aggregates three sparse branches with
learned gates (Eq. 5):

    o* = sum_{c in {cmp, slc, win}} g_c * Attn(q, K~_c, V~_c)

  - Compression (cmp): contiguous token blocks are collapsed into a single
    token each by a learnable MLP `phi` with intra-block position encoding,
    giving coarse global context (Eq. 7).
  - Selection  (slc): block importance is read off the compression attention
    scores (Eq. 8-10); the top-n most relevant blocks are kept and attended to
    at full token granularity (Eq. 11-12).
  - Sliding window (win): a local branch isolates short-range context so it
    cannot shortcut the cmp/slc branches (Sec. 3.3.3).
  - Each branch owns independent keys/values; gates g_c = sigmoid(MLP(x)).

This file keeps DEIMv2's decoder contract intact: cross-attention stays as
MSDeformableAttention (it owns multi-scale reference-point sampling), and only
the query self-attention is swapped. Object queries have no temporal order, so
following the LWGA decoder convention tokens are sorted by their reference
points before the block / window operations, then restored to the original
query order. Contrastive-denoising reconstruction groups are processed
independently so the DN attention mask is respected; chunks too small to block
fall back to a plain MHA.
"""

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core import register
from .deim_decoder import DEIMTransformer
from .dfine_decoder import MSDeformableAttention, LQE
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate
from .utils import inverse_sigmoid

__all__ = ['DEIMTransformer_NSA']


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, N, C) -> (B, H, N, d_h)."""
    b, n, c = x.shape
    d_h = c // num_heads
    return x.view(b, n, num_heads, d_h).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """(B, H, N, d_h) -> (B, N, C)."""
    b, h, n, d_h = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * d_h)


class NSABranchAttention(nn.Module):
    """Multi-head attention helper with independent per-branch K/V/out.

    The query projection is shared across the three branches (owned by the
    enclosing module and passed in already projected), since all branches
    attend from the same query set. Each branch keeps its own k/v/out so the
    keys/values remain independent (NSA Sec. 3.3.3 - "independent keys and
    values for three branches").

    Math is kept explicit (no FlashAttention dependency) so the branches stay
    natively trainable end-to-end on the small object-query sequence.
    """

    def __init__(self, dim: int, num_heads: int, out_low_rank: Optional[int] = None):
        super().__init__()
        assert dim % num_heads == 0, 'NSA dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # Low-rank output projection: Linear(dim, r) -> GELU -> Linear(r, dim).
        # This shrinks the per-branch out parameters (dim^2 -> 2*dim*r) without
        # touching the attention math, so NSA's algorithm is unchanged. r=None
        # falls back to a full Linear.
        if out_low_rank is not None and out_low_rank < dim:
            self.out_proj = nn.Sequential(
                nn.Linear(dim, out_low_rank), nn.GELU(), nn.Linear(out_low_rank, dim))
        else:
            self.out_proj = nn.Linear(dim, dim)

    def forward(self,
                q_heads: Tensor,
                k_in: Tensor,
                v_in: Tensor,
                attn_mask: Optional[Tensor] = None,
                return_scores: bool = False,
                gate: Optional[Tensor] = None):
        """q_heads: (B, H, Nq, d) already projected+split by the caller;
        k_in/v_in: (B, Nkv, C). Returns (B, Nq, C).

        attn_mask is broadcastable to (B, H, Nq, Nkv) and boolean; True
        positions are ignored. `gate` is the optional G1 gate logits
        ((B,H,Nq,1) or (B,H,Nq,d)); when given the attention output is
        multiplied by sigmoid(gate) before out_proj.
        """
        q = q_heads                                                      # (B,H,Nq,d)
        k = _split_heads(self.k_proj(k_in), self.num_heads)            # (B,H,Nkv,d)
        v = _split_heads(self.v_proj(v_in), self.num_heads)            # (B,H,Nkv,d)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # (B,H,Nq,Nkv)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)                                    # (B,H,Nq,d)
        if gate is not None:
            out = out * torch.sigmoid(gate)                            # G1 gate
        out = _merge_heads(out)
        if return_scores:
            # Return the softmax weights so the selection branch can reuse them
            # as block-importance scores (NSA Eq. 8-9).
            return self.out_proj(out), attn
        return self.out_proj(out)

    def band_window_forward(self,
                            q_heads: Tensor,
                            k_in: Tensor,
                            v_in: Tensor,
                            window_size: int,
                            gate: Optional[Tensor] = None) -> Tensor:
        """Sliding-window (local) attention with O(N * w) cost.

        The DEIM decoder queries carry no temporal order, so the caller first
        sorts them by reference point; on that sorted sequence, "window" means
        each query attends to itself and its `window_size` nearest neighbours
        on each side. Rather than materialising a full N x N score matrix and
        masking it (O(N^2) memory + FLOPs), we process the sequence in chunks
        of `window_size`: each chunk attends to itself plus the previous chunk
        (2 * window_size tokens), giving each query exactly a window of width
        ~2 * window_size at O(N * w * d) cost.

        q_heads: (B, H, N, d) already projected+split by the caller.
        k_in/v_in: (B, N, C). `gate` is the optional G1 logits. Returns (B, N, C).
        """
        b, h, n, d = q_heads.shape
        c = h * d
        w = int(window_size)
        # When the sequence is shorter than one window, band == full attention.
        if n <= w:
            return self.forward(q_heads, k_in, v_in, attn_mask=None, gate=gate)

        k = _split_heads(self.k_proj(k_in), self.num_heads)            # (B,H,N,d)
        v = _split_heads(self.v_proj(v_in), self.num_heads)            # (B,H,N,d)

        # Pad: prepend one zero-window (so the first real chunk has a left
        # neighbour to attend to) and append enough zeros to make the total
        # length a multiple of w.
        tail = (w - (n % w)) % w
        def z(t):
            return torch.zeros(b, h, t, d, device=q_heads.device, dtype=q_heads.dtype)
        q_pad = torch.cat([z(w), q_heads, z(tail)], dim=2)            # (B,H,w+n+tail,d)
        k_pad = torch.cat([z(w), k, z(tail)], dim=2)
        v_pad = torch.cat([z(w), v, z(tail)], dim=2)

        m = q_pad.shape[2]
        nch = m // w
        qc = q_pad.view(b, h, nch, w, d)                              # (B,H,nch,w,d)
        kc = k_pad.view(b, h, nch, w, d)
        vc = v_pad.view(b, h, nch, w, d)

        # Each real chunk (1..nch-1) attends to itself + the previous chunk.
        q_win = qc[:, :, 1:]                                          # (B,H,nch-1,w,d)
        k_win = torch.cat([kc[:, :, :-1], kc[:, :, 1:]], dim=3)       # (B,H,nch-1,2w,d)
        v_win = torch.cat([vc[:, :, :-1], vc[:, :, 1:]], dim=3)       # (B,H,nch-1,2w,d)

        scores = torch.einsum('bhcwd,bhckd->bhcwk', q_win, k_win) * self.scale  # (B,H,nch-1,w,2w)
        attn = scores.softmax(dim=-1)
        out = torch.einsum('bhcwk,bhckd->bhcwd', attn, v_win)         # (B,H,nch-1,w,d)

        out = out.reshape(b, h, (nch - 1) * w, d)
        out = out[:, :, :n].contiguous()                              # drop leading zero-window
        if gate is not None:
            out = out * torch.sigmoid(gate)                           # G1 gate (B,H,N,d)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.out_proj(out)

    def project_kv_heads(self, kv_in: Tensor) -> Tuple[Tensor, Tensor]:
        """Project a shared KV set into head space once.

        kv_in: (B, Nkv, C) -> (k_heads, v_heads), each (B, H, Nkv, d). Used by
        the selection branch so the K/V projection is computed once over all
        tokens (NSA-style shared projection) instead of once per query.
        """
        k = _split_heads(self.k_proj(kv_in), self.num_heads)            # (B,H,Nkv,d)
        v = _split_heads(self.v_proj(kv_in), self.num_heads)            # (B,H,Nkv,d)
        return k, v

    def per_query_gather_forward(self,
                                 q_heads: Tensor,
                                 kv_heads_full: Tuple[Tensor, Tensor],
                                 block_idx: Tensor,
                                 block_len: int,
                                 gate: Optional[Tensor] = None) -> Tensor:
        """Shared-projection per-query attention for the selection branch.

        The K/V are projected ONCE over all tokens (shared projection, NSA
        style) into `kv_heads_full`, then each query gathers its own top-n
        blocks from the already-projected heads. This avoids re-projecting the
        per-query KV set, which dominates selection FLOPs.

        Args:
            q_heads: (B, H, Nq, d) already projected+split by the caller.
            kv_heads_full: (k, v), each (B, H, Ntok, d) - projection of all
                block tokens (Ntok = num_blocks * block_len).
            block_idx: (B, Nq, n) selected block indices per query.
            block_len: l, tokens per block.
            gate: optional G1 logits ((B,H,Nq,1) or (B,H,Nq,d)).
        Returns:
            (B, Nq, C).
        """
        b, h, nq, d = q_heads.shape
        c = h * d
        k_full, v_full = kv_heads_full                       # (B,H,Ntok,d) each
        ntok = k_full.shape[2]

        # Expand block indices to token indices: (B, Nq, n, l) -> (B, Nq, n*l).
        n = block_idx.shape[2]
        tok_off = torch.arange(block_len, device=block_idx.device)
        tok_idx = block_idx.unsqueeze(-1) * block_len + tok_off.view(1, 1, 1, block_len)
        tok_idx = tok_idx.reshape(b, nq, n * block_len)      # (B, Nq, n*l)

        # Gather projected heads per query. Build an index of shape
        # (B, H, Nq*n*l, 1) and gather along the token axis of (B,H,Ntok,d),
        # then reshape to (B, H, Nq, n*l, d).
        sel = n * block_len
        idx = tok_idx.reshape(b, 1, nq * sel, 1).expand(b, h, nq * sel, d)   # (B,H,Nq*n*l,d)
        k_sel = torch.gather(k_full, 2, idx).view(b, h, nq, sel, d)           # (B,H,Nq,n*l,d)
        v_sel = torch.gather(v_full, 2, idx).view(b, h, nq, sel, d)

        q = q_heads.unsqueeze(-2)                            # (B,H,Nq,1,d)
        scores = (q * k_sel).sum(dim=-1) * self.scale        # (B,H,Nq,n*l)
        attn = scores.softmax(dim=-1)
        out = (attn.unsqueeze(-1) * v_sel).sum(dim=-2)       # (B,H,Nq,d)
        if gate is not None:
            out = out * torch.sigmoid(gate)                  # G1 gate
        out = out.transpose(1, 2).contiguous().view(b, nq, c)
        return self.out_proj(out)


class NSACompression(nn.Module):
    """Token-compression branch (NSA Sec. 3.3.1).

    Collapses each contiguous block of `block_len` tokens into a single
    compressed token, producing compact K~_cmp / V~_cmp.

    The original NSA `phi` flattens a block (block_len * dim) into a big MLP.
    That MLP dominates the layer's parameter/FLOP budget, so here a lighter
    equivalent is used: add intra-block position encodings, mean-pool each
    block to one dim-vector, then project with a single shared Linear `phi`.
    One compressed token per block is produced; the K/V distinction is left to
    the branch's own independent k_proj/v_proj (NSA keeps branch K/V
    independent), so sharing the compressor itself does not violate the
    "independent keys/values" design. This preserves the "block -> one
    compressed token" semantics at a small fraction of the cost.
    """

    def __init__(self, dim: int, block_len: int):
        super().__init__()
        self.block_len = block_len
        self.pos = nn.Parameter(torch.zeros(block_len, dim))
        # phi: project the mean-pooled block vector to one compressed token.
        self.phi = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.xavier_uniform_(self.phi.weight)

    def forward(self, x: Tensor):
        """x: (B, N, C) -> compressed tokens (B, M, C).

        The caller (cmp_attn) applies its own independent k_proj/v_proj to turn
        this shared compressed representation into K~_cmp / V~_cmp.
        """
        b, n, c = x.shape
        l = self.block_len
        num_blocks = n // l
        if num_blocks == 0:
            # Fewer tokens than one block: degrade to a single compressed token
            # padded with zeros so the branch still emits a valid KV set.
            pad = torch.zeros(b, l - n, c, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            num_blocks = 1
        blocks = x[:, :num_blocks * l].view(b, num_blocks, l, c)
        blocks = blocks + self.pos  # intra-block position encoding
        pooled = blocks.mean(dim=2)               # (B, M, C)
        return self.phi(pooled)


class NSATokenSelfAttention(nn.Module):
    """NSA self-attention over decoder query tokens (B, N, C) -> (B, N, C).

    Implements the three-branch sparse attention of NSA with per-branch
    independent K/V and learned gating, adapted to bidirectional set
    self-attention (no causal mask) used by the DEIM decoder.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dropout: float = 0.,
                 block_len: int = 8,
                 num_selected_blocks: int = 4,
                 window_size: int = 8,
                 min_tokens: int = 16,
                 sort_by_ref: bool = True,
                 isolate_dn: bool = True,
                 out_low_rank: Optional[int] = None,
                 attn_gate: str = 'none'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_len = int(block_len)
        self.num_selected_blocks = int(num_selected_blocks)
        self.window_size = int(window_size)
        self.min_tokens = int(min_tokens)
        self.sort_by_ref = bool(sort_by_ref)
        self.isolate_dn = bool(isolate_dn)

        # Gated Attention (G1) - a query-dependent sparse gate applied to each
        # branch's attention output BEFORE out_proj (Qiu et al., arXiv:2505.06708).
        # The gate logits are produced by over-projecting the shared query, so
        # no extra module is added. Shared across the three NSA branches.
        #   'none'       : no G1 gate (plain NSA).
        #   'headwise'   : one scalar gate per (query, head) -> (B,H,N,1).
        #   'elementwise': one gate per (query, head, head_dim) -> (B,H,N,d).
        self.attn_gate = str(attn_gate).lower()
        assert self.attn_gate in ('none', 'headwise', 'elementwise'), \
            f'nsa_attn_gate must be one of none/headwise/elementwise, got {attn_gate}'
        if self.attn_gate == 'headwise':
            gate_dim = num_heads                # 1 logit per head
        elif self.attn_gate == 'elementwise':
            gate_dim = num_heads * self.head_dim  # head_dim logits per head
        else:
            gate_dim = 0

        # Shared query projection for all three branches (all attend from the
        # same query set); each branch below keeps its own K/V/out. When a G1
        # gate is enabled, q_proj is over-projected to also emit the gate logits.
        self.q_proj = nn.Linear(dim, dim + gate_dim)
        self.gate_dim = gate_dim

        # Low-rank output projection rank (default dim//4). Each branch keeps
        # its own out_proj (independent branches), only the projection is shrunk.
        if out_low_rank is None:
            out_low_rank = max(1, dim // 4)

        # Compression branch: shared compressor + its own k/v/out projections.
        self.compress = NSACompression(dim, self.block_len)
        self.cmp_attn = NSABranchAttention(dim, num_heads, out_low_rank=out_low_rank)

        # Selection branch: attends to top-n selected blocks at token granularity.
        self.slc_attn = NSABranchAttention(dim, num_heads, out_low_rank=out_low_rank)

        # Sliding-window branch: local banded attention. With no mask it
        # degenerates to full attention, which is also how chunks too small to
        # block (e.g. tiny DN reconstruction groups) are handled.
        self.win_attn = NSABranchAttention(dim, num_heads, out_low_rank=out_low_rank)

        # Learned per-query gates g_c (NSA Eq. 5), independent for the 3 branches.
        # Single linear layer: sigmoid(0) = 0.5 gives a balanced start.
        self.gate = nn.Linear(dim, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        for m in (self.cmp_attn, self.slc_attn, self.win_attn):
            nn.init.xavier_uniform_(m.k_proj.weight)
            nn.init.constant_(m.k_proj.bias, 0.)
            nn.init.xavier_uniform_(m.v_proj.weight)
            nn.init.constant_(m.v_proj.bias, 0.)
            # out_proj may be a low-rank Sequential (Linear-GELU-Linear).
            for mod in m.out_proj.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.constant_(mod.bias, 0.)
        # Near-zero gate logits -> ~balanced branches at start, still trainable.
        nn.init.normal_(self.gate.weight, std=1e-2)
        nn.init.zeros_(self.gate.bias)

    def project_query(self, x: Tensor):
        """Over-project the shared query and split off the G1 gate logits.

        Returns (q_heads, gate) where q_heads is (B, H, N, d) for attention and
        gate is None when attn_gate == 'none', else (B, H, N, 1) for headwise or
        (B, H, N, d) for elementwise - the G1 gate logits to apply to each
        branch's attention output.
        """
        q = self.q_proj(x)                                   # (B, N, dim + gate_dim)
        if self.gate_dim == 0:
            return _split_heads(q, self.num_heads), None     # (B,H,N,d), None
        qh, gate = q.split([self.dim, self.gate_dim], dim=-1)
        qh = _split_heads(qh, self.num_heads)                # (B,H,N,d)
        b, h, n, d = qh.shape
        if self.attn_gate == 'headwise':
            gate = gate.view(b, n, self.num_heads).permute(0, 2, 1).contiguous()  # (B,H,N)
            gate = gate.unsqueeze(-1)                        # (B,H,N,1)
        else:  # elementwise
            gate = gate.view(b, n, self.num_heads, d).permute(0, 2, 1, 3).contiguous()  # (B,H,N,d)
        return qh, gate

    # ------------------------------------------------------------------
    # ordering / denoising helpers (mirror the LWGA decoder convention)
    # ------------------------------------------------------------------
    def _order_from_ref(self, reference_points: Optional[Tensor], n_tokens: int):
        if reference_points is None or not self.sort_by_ref:
            return None, None
        ref = reference_points
        if ref.dim() == 4:
            ref = ref[:, :, 0, :]
        xy = ref[:, :n_tokens, :2].detach().clamp(0, 1)
        scores = xy[..., 1] * 1024.0 + xy[..., 0]
        order = torch.argsort(scores, dim=1)
        inv_order = torch.argsort(order, dim=1)
        return order, inv_order

    def _split_with_dn(self, tokens: Tensor, reference_points: Optional[Tensor], dn_meta):
        dn_split = dn_meta.get('dn_num_split', None) if isinstance(dn_meta, dict) else None
        if not self.isolate_dn or dn_split is None or dn_split[0] == 0:
            return [(tokens, reference_points)]

        dn_tokens, match_tokens = dn_split
        num_group = max(int(dn_meta.get('dn_num_group', 1)), 1)
        group_size = dn_tokens // num_group

        chunks = []
        for i in range(num_group):
            start = i * group_size
            end = dn_tokens if i == num_group - 1 else (i + 1) * group_size
            ref = None if reference_points is None else reference_points[:, start:end]
            chunks.append((tokens[:, start:end], ref))

        ref = None if reference_points is None else reference_points[:, dn_tokens:dn_tokens + match_tokens]
        chunks.append((tokens[:, dn_tokens:dn_tokens + match_tokens], ref))
        return chunks

    # ------------------------------------------------------------------
    # NSA core
    # ------------------------------------------------------------------
    def _block_importance(self, p_cmp: Tensor) -> Tensor:
        """Aggregate per-head compression scores into shared block importance.

        p_cmp: (B, H, N, M) softmax weights from the compression branch.
        Returns (B, N, M). Mirrors NSA Eq. 10 (importance shared across heads).
        """
        return p_cmp.mean(dim=1)

    def _select_topn_blocks(self, importance: Tensor, num_blocks: int):
        """importance: (B, N, M) -> selected block indices (B, N, n_eff)."""
        n = min(self.num_selected_blocks, num_blocks)
        _, idx = torch.topk(importance, n, dim=-1)  # (B, N, n)
        return idx, n

    def _apply_nsa(self, tokens: Tensor, reference_points: Optional[Tensor]) -> Tensor:
        """Run NSA on one (already DN-isolated) chunk. Returns the update."""
        b, n, c = tokens.shape
        if n == 0:
            return tokens

        order, inv_order = self._order_from_ref(reference_points, n)
        if order is not None:
            x = tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, c))
        else:
            x = tokens

        # Project the query once and share it across all three branches. When
        # G1 gated attention is enabled, this also over-projects the gate logits.
        q_heads, g1_gate = self.project_query(x)                  # (B,H,N,d), gate?

        # Too few tokens to block meaningfully: run the window branch without a
        # mask (== full self-attention) so tiny DN reconstruction groups still
        # train. No separate dense module is needed.
        if n < self.min_tokens:
            out = self.win_attn(q_heads, x, x, attn_mask=None, gate=g1_gate)
            if inv_order is not None:
                return out.gather(1, inv_order.unsqueeze(-1).expand(-1, -1, c))
            return out

        l = self.block_len
        num_blocks = n // l  # non-overlapping blocks (M)

        # ---- Compression branch (Eq. 7) ----
        # compress() yields one shared compressed token per block; the branch's
        # own independent k_proj/v_proj turn it into K~_cmp / V~_cmp.
        cmp_tokens = self.compress(x)                    # (B, M, C)
        o_cmp, p_cmp = self.cmp_attn(q_heads, cmp_tokens, cmp_tokens,
                                     return_scores=True, gate=g1_gate)

        # ---- Selection branch (Eq. 8-12) ----
        # Each query selects its own top-n blocks. To avoid re-projecting the
        # per-query KV set (which dominates selection FLOPs), project the K/V
        # ONCE over all block tokens (NSA-style shared projection), then let
        # each query gather its selected blocks from the projected heads.
        importance = self._block_importance(p_cmp)       # (B, N, M)
        sel_idx, n_sel = self._select_topn_blocks(importance, num_blocks)
        block_tokens = x[:, :num_blocks * l].reshape(b, num_blocks * l, c)  # (B, M*l, C)
        slc_kv_heads = self.slc_attn.project_kv_heads(block_tokens)          # (k,v) each (B,H,M*l,d)
        o_slc = self.slc_attn.per_query_gather_forward(
            q_heads, slc_kv_heads, sel_idx, l, gate=g1_gate)

        # ---- Sliding-window branch (Sec. 3.3.3) ----
        # Local band attention on the reference-sorted sequence: each query
        # attends to its `window_size` nearest neighbours per side, computed at
        # O(N * w) cost instead of a full N x N matrix (which the old mask
        # degenerated into at N=300).
        o_win = self.win_attn.band_window_forward(q_heads, x, x, self.window_size,
                                                 gate=g1_gate)

        # ---- Gated aggregation (Eq. 5) ----
        gates = self.gate(x).sigmoid()                   # (B, N, 3)
        g_cmp, g_slc, g_win = gates.unbind(-1)          # (B, N) each
        out = g_cmp.unsqueeze(-1) * o_cmp + \
              g_slc.unsqueeze(-1) * o_slc + \
              g_win.unsqueeze(-1) * o_win

        if inv_order is not None:
            return out.gather(1, inv_order.unsqueeze(-1).expand(-1, -1, c))
        return out

    def forward(self, target: Tensor, query_pos_embed: Optional[Tensor],
                reference_points: Optional[Tensor], attn_mask=None, dn_meta=None) -> Tensor:
        tokens = target if query_pos_embed is None else target + query_pos_embed

        # An attn_mask without denoising context (not produced by the DEIM
        # decoder in practice) cannot be represented by the sparse branches, so
        # honour it explicitly via a masked full-attention pass using the window
        # branch. DN masks are instead honoured by processing each
        # reconstruction group independently below.
        if attn_mask is not None and dn_meta is None:
            b, n, c = tokens.shape
            q_heads, g1_gate = self.project_query(tokens)
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.view(1, 1, n, n)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            return self.win_attn(q_heads, tokens, tokens, attn_mask=mask, gate=g1_gate)

        chunks = self._split_with_dn(tokens, reference_points, dn_meta)
        if len(chunks) == 1:
            return self._apply_nsa(tokens, reference_points)
        return torch.cat([self._apply_nsa(x, ref) for x, ref in chunks], dim=1)


class TransformerDecoderLayer_NSA(nn.Module):
    """DEIM decoder layer with NSA replacing query self-attention."""

    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_gateway=False,
                 nsa_block_len=8,
                 nsa_num_selected_blocks=4,
                 nsa_window_size=8,
                 nsa_min_tokens=16,
                 nsa_sort_by_ref=True,
                 nsa_isolate_dn=True,
                 nsa_out_low_rank=None,
                 nsa_attn_gate='none',
                 ):
        super().__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = NSATokenSelfAttention(
            d_model,
            n_head,
            dropout=dropout,
            block_len=nsa_block_len,
            num_selected_blocks=nsa_num_selected_blocks,
            window_size=nsa_window_size,
            min_tokens=nsa_min_tokens,
            sort_by_ref=nsa_sort_by_ref,
            isolate_dn=nsa_isolate_dn,
            out_low_rank=nsa_out_low_rank,
            attn_gate=nsa_attn_gate,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None,
                dn_meta=None):

        target2 = self.self_attn(target, query_pos_embed, reference_points, attn_mask, dn_meta)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder_NSA(nn.Module):
    """Transformer decoder copy that passes dn_meta into NSA self-attention."""

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
                                    + [copy.deepcopy(decoder_layer_wide)
                                       for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act))
                                         for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, 'project') else self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes,
                           attn_mask, query_pos_embed, dn_meta=dn_meta)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
            torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


@register()
class DEIMTransformer_NSA(DEIMTransformer):
    """DEIMTransformer with NSA query self-attention in each decoder layer."""

    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 nsa_block_len=8,
                 nsa_num_selected_blocks=4,
                 nsa_window_size=8,
                 nsa_min_tokens=16,
                 nsa_sort_by_ref=True,
                 nsa_isolate_dn=True,
                 nsa_out_low_rank=None,
                 nsa_attn_gate='none',
                 ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
            use_gateway=use_gateway,
            share_bbox_head=share_bbox_head,
            share_score_head=share_score_head,
        )

        decoder_layer = TransformerDecoderLayer_NSA(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
            nsa_block_len=nsa_block_len,
            nsa_num_selected_blocks=nsa_num_selected_blocks,
            nsa_window_size=nsa_window_size,
            nsa_min_tokens=nsa_min_tokens,
            nsa_sort_by_ref=nsa_sort_by_ref,
            nsa_isolate_dn=nsa_isolate_dn,
            nsa_out_low_rank=nsa_out_low_rank,
            nsa_attn_gate=nsa_attn_gate,
        )
        decoder_layer_wide = TransformerDecoderLayer_NSA(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
            nsa_block_len=nsa_block_len,
            nsa_num_selected_blocks=nsa_num_selected_blocks,
            nsa_window_size=nsa_window_size,
            nsa_min_tokens=nsa_min_tokens,
            nsa_sort_by_ref=nsa_sort_by_ref,
            nsa_isolate_dn=nsa_isolate_dn,
            nsa_out_low_rank=nsa_out_low_rank,
            nsa_attn_gate=nsa_attn_gate,
        )
        self.decoder = TransformerDecoder_NSA(
            hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
            reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
