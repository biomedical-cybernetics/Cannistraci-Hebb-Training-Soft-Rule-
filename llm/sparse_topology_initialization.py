from __future__ import annotations

from typing import Any

import numpy as np
import torch


def create_ws_sparse_scheduler(sparsity: float, w: torch.Tensor, args: Any) -> torch.Tensor:
    """
    Watts–Strogatz-style bipartite initialization mask for a Linear weight matrix.

    This builds a structured local connectivity pattern and then rewires a fraction
    of existing edges with probability `args.ws_beta`, keeping the total number of
    active edges approximately constant.

    Args:
        sparsity: Fraction of zeros in the mask, in [0, 1).
        w: Weight tensor (2D) whose shape the mask should match (or transpose to match).
        args: Should provide `ws_beta` (float in [0,1]).

    Returns:
        A Long tensor mask (0/1) on the same device as `w`, shaped like `w`.
    """
    if w.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(w.shape)}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"'sparsity' must be in [0, 1). Got {sparsity}")

    ws_beta = float(getattr(args, "ws_beta", 0.0))
    if not (0.0 <= ws_beta <= 1.0):
        raise ValueError(f"'ws_beta' must be in [0, 1]. Got {ws_beta}")

    device = w.device
    in_dim = min(w.shape[0], w.shape[1])
    out_dim = max(w.shape[0], w.shape[1])

    # Desired average degree per output column (in_dim neighbors)
    k_float = (1.0 - sparsity) * in_dim
    if k_float <= 0:
        # fully sparse
        mask = torch.zeros((in_dim, out_dim), dtype=torch.long, device=device)
        return mask.t() if w.shape[0] != in_dim else mask

    k1, k2 = int(np.floor(k_float)), int(np.floor(k_float)) + 1

    # Distribute degrees so that the mean is ~k_float
    # number of columns with degree k2 is approximately (k_float - k1) * out_dim
    n_k2 = int(round((k_float - k1) * out_dim))
    n_k2 = max(0, min(out_dim, n_k2))
    degs = np.array([k2] * n_k2 + [k1] * (out_dim - n_k2), dtype=np.int64)
    np.random.shuffle(degs)

    # Build initial ring-like adjacency: for each output column j, connect to a local window in input space.
    adj = np.zeros((in_dim, out_dim), dtype=np.int8)
    rate = in_dim / out_dim
    for j in range(out_dim):
        kj = int(degs[j])
        if kj <= 0:
            continue
        start = int(j * rate - kj / 2)
        idx = (start + np.arange(kj)) % in_dim
        adj[idx, j] = 1

    # Rewiring: drop a subset of existing edges, then add same number of random new edges.
    if ws_beta > 0.0:
        ones = np.argwhere(adj == 1)  # shape: (nnz, 2)
        nnz = ones.shape[0]
        if nnz > 0:
            drop = np.random.rand(nnz) < ws_beta
            dropped = ones[drop]
            if dropped.size > 0:
                adj[dropped[:, 0], dropped[:, 1]] = 0

            # Regrow exactly the number dropped (attempts bounded to avoid infinite loops)
            to_add = int(dropped.shape[0])
            if to_add > 0:
                zeros = np.argwhere(adj == 0)
                if zeros.shape[0] < to_add:
                    # Very unlikely unless near-dense; fallback: add as many as possible.
                    to_add = zeros.shape[0]
                choice = np.random.choice(zeros.shape[0], size=to_add, replace=False)
                picked = zeros[choice]
                adj[picked[:, 0], picked[:, 1]] = 1

    mask = torch.from_numpy(adj).to(device=device, dtype=torch.long)
    return mask.t() if w.shape[0] != in_dim else mask


def create_brf_sparse_scheduler(sparsity: float, w: torch.Tensor, args: Any) -> torch.Tensor:
    """
    BRF (Bipartite Receptive Field) initialization mask.

    Each input row i connects to Ki output columns, where columns are chosen either:
      - deterministically (delta=0): closest columns on a ring
      - probabilistically (delta>0): sample with p ∝ (1 + ring_distance)^(-alpha),
        where alpha = (1-delta)/delta.

    Args:
        sparsity: Fraction of zeros in the mask, in [0, 1).
        w: Weight tensor (2D).
        args: Should provide `brf_r` (delta in [0,1]) and optionally `degree_dist` ("uniform").

    Returns:
        A Long tensor mask (0/1) on the same device as `w`, shaped like `w`.
    """
    if w.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(w.shape)}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"'sparsity' must be in [0, 1). Got {sparsity}")

    delta = float(getattr(args, "brf_r", 0.0))
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"'brf_r' (delta) must be in [0, 1]. Got {delta}")

    device = w.device
    in_dim = min(w.shape[0], w.shape[1])
    out_dim = max(w.shape[0], w.shape[1])

    # Per-row expected degree (in output space), total links scaled by in_dim.
    k_float = (1.0 - sparsity) * out_dim
    total_links = int(round(k_float * in_dim))
    if total_links <= 0:
        mask = torch.zeros((in_dim, out_dim), dtype=torch.long, device=device)
        return mask.t() if w.shape[1] != out_dim else mask

    # Build per-row degrees that sum to total_links.
    degree_dist = getattr(args, "degree_dist", None)
    if degree_dist == "uniform":
        if total_links < in_dim:
            raise ValueError("Total links < number of rows; cannot give each row at least degree 1.")
        row_degrees = np.ones(in_dim, dtype=np.int64)
        rem = total_links - in_dim
        if rem > 0:
            add_rows = np.random.randint(0, in_dim, size=rem)
            np.add.at(row_degrees, add_rows, 1)
    else:
        k1 = int(np.floor(k_float))
        frac = float(k_float - k1)
        n_k2 = int(round(frac * in_dim))
        n_k2 = max(0, min(in_dim, n_k2))
        row_degrees = np.array([k1 + 1] * n_k2 + [k1] * (in_dim - n_k2), dtype=np.int64)
        np.random.shuffle(row_degrees)

    # Distance exponent
    alpha = None if delta == 0.0 else (1.0 - delta) / max(delta, 1e-12)

    adj = np.zeros((in_dim, out_dim), dtype=np.int8)
    stride = max(1, out_dim // max(1, in_dim))

    for i in range(in_dim):
        ki = int(row_degrees[i])
        if ki <= 0:
            continue
        ki = min(ki, out_dim)  # cannot sample more than out_dim without replacement

        center = (i * stride) % out_dim
        d = np.abs(np.arange(out_dim) - center)
        d = np.minimum(d, out_dim - d)  # ring distance

        if delta == 0.0:
            chosen = np.argpartition(d, kth=min(ki - 1, out_dim - 1))[:ki]
        else:
            weights = (1.0 + d) ** (-alpha)
            weights = weights / weights.sum()
            chosen = np.random.choice(out_dim, size=ki, replace=False, p=weights)

        adj[i, chosen] = 1

    mask = torch.from_numpy(adj).to(device=device, dtype=torch.long)
    return mask.t() if w.shape[1] != out_dim else mask
