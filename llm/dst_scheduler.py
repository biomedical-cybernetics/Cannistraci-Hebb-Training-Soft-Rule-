from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import csr_matrix

from dst_util import get_W
from sparse_topology_initialization import (
    create_brf_sparse_scheduler,
    create_ws_sparse_scheduler,
)

# Optional dependency (used only for CH*_L3p regrowth)
try:
    import CH_scores  # type: ignore
except Exception:
    CH_scores = None


Mask = torch.Tensor  # expected dtype: torch.bool


# -----------------------------
# Mask utilities (chain pruning)
# -----------------------------

def _remove_inactive_backward(current: Mask, after: Mask) -> Mask:
    """
    Remove links in `current` whose target neurons have 0 outdegree in `after`.
    Shapes:
      current: (out, in)
      after:   (next_out, out) or compatible so that after.sum(dim=0) matches current.shape[0]
    """
    # outdegree over "out" dimension of current
    out_active = after.sum(dim=0) > 0  # (out,)
    return current & out_active.view(-1, 1)


def _remove_inactive_forward(current: Mask, before: Mask) -> Mask:
    """
    Remove links in `current` whose source neurons have 0 indegree in `before`.
    Shapes:
      before: (out, in) so that before.sum(dim=1) matches current.shape[1]
      current: (out, in)
    """
    in_active = before.sum(dim=1) > 0  # (out,)
    return current & in_active.view(1, -1)


def chain_removal(mask1: Mask, mask2: Mask) -> Tuple[Mask, Mask]:
    """
    Enforce activity consistency between two consecutive adjacency masks.
    """
    mask1 = _remove_inactive_backward(mask1, mask2)
    mask2 = _remove_inactive_forward(mask2, mask1)
    return mask1, mask2


def qk_chain_removal(q: Mask, k: Mask) -> Tuple[Mask, Mask]:
    """
    Enforce consistency for Q/K projections (note transpose use).
    """
    q = _remove_inactive_backward(q, k.transpose(1, 0))
    k = _remove_inactive_backward(k, q.transpose(1, 0))
    return q, k


# -----------------------------
# Backward hook (mask gradients)
# -----------------------------

class IndexMaskHook:
    """
    Gradient hook that:
      1) masks gradients so pruned weights never get updates,
      2) optionally accumulates a rolling dense gradient estimate for regrowth.
    """

    def __init__(self, layer_idx: int, scheduler: "DSTScheduler") -> None:
        self.layer_idx = layer_idx
        self.scheduler = scheduler
        self.dense_grad: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return f"IndexMaskHook(layer_idx={self.layer_idx})"

    @torch.no_grad()
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        mask = self.scheduler.backward_masks[self.layer_idx]
        if mask is None:
            return grad

        if self.scheduler.should_accumulate_dense_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad.add_(grad, alpha=1.0 / self.scheduler.grad_accumulation_n)
        else:
            self.dense_grad = None

        return grad * mask


def _wrap_optimizer_step(scheduler: "DSTScheduler", optimizer: torch.optim.Optimizer) -> None:
    """
    Wrap optimizer.step() to enforce:
      - momentum masking
      - weight masking
    after every step.

    Note: this mutates optimizer.step.
    """
    if getattr(scheduler.args, "ssam", False):
        base_step = optimizer._optimizer.second_step  # type: ignore[attr-defined]
        def wrapped_step() -> None:
            base_step(zero_grad=False)
            scheduler.reset_momentum()
            scheduler.apply_mask_to_weights()
    else:
        base_step = optimizer.step
        def wrapped_step(*args, **kwargs) -> None:  # keep signature flexible
            base_step(*args, **kwargs)
            scheduler.reset_momentum()
            scheduler.apply_mask_to_weights()

    optimizer.step = wrapped_step  # type: ignore[assignment]


# -----------------------------
# DST Scheduler
# -----------------------------

class DSTScheduler:
    """
    Dynamic Sparse Training scheduler operating on selected LLM projection matrices.

    Core responsibilities:
      - initialize sparse masks (random / WS / BRF)
      - enforce masks on weights/gradients
      - perform periodic remove + regrow steps
      - optional gradual pruning (GraNet / GMP)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        T_end: Optional[int] = None,
        sparsity_distribution: str = "uniform",
        ignore_linear_layers: bool = True,
        delta: int = 100,
        alpha: float = 0.3,
        static_topo: bool = False,
        grad_accumulation_n: int = 1,
        state_dict: Optional[Dict[str, Any]] = None,
        args: Any = None,
    ) -> None:
        if args is None:
            raise ValueError("DSTScheduler requires `args` with sparsity/pruning/regrow configuration.")

        self.args = args
        self.model = model
        self.optimizer = optimizer

        self.is_dist = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.rank = dist.get_rank() if self.is_dist else 0

        self.sparsity_distribution = sparsity_distribution
        self.ignore_linear_layers = ignore_linear_layers
        self.static_topo = static_topo
        self.grad_accumulation_n = int(grad_accumulation_n)
        if self.grad_accumulation_n <= 0:
            raise ValueError("grad_accumulation_n must be > 0")

        # derive dense allocation
        init_sp = getattr(self.args, "granet_init_sparsity", None)
        target_sp = getattr(self.args, "sparsity", None)
        use_granet = bool(getattr(self.args, "granet", False) or getattr(self.args, "gmp", False))

        if use_granet:
            if init_sp is None:
                raise ValueError("args.granet_init_sparsity must be set when using granet/gmp.")
            self.dense_allocation = 1.0 - float(init_sp)
        else:
            if target_sp is None:
                raise ValueError("args.sparsity must be set.")
            self.dense_allocation = 1.0 - float(target_sp)

        if not (0.0 < self.dense_allocation <= 1.0):
            raise ValueError(f"dense_allocation must be in (0,1]. Got {self.dense_allocation}")

        self.global_sparsity = 1.0 - self.dense_allocation

        # collect weights and chain relations
        self.W, self.chain_list, self.qk_chain_list = get_W(model, args)
        self.layer_meta = self._build_layer_meta(model)

        self.N = [w.numel() for w in self.W]

        # scheduling params
        self.delta_T = int(delta)
        self.alpha = float(alpha)
        self.T_end = int(T_end) if T_end is not None else int(getattr(self.args, "T_end", 0) or 0)

        # pruning window (GraNet/GMP)
        self.pruning_T_end = int(getattr(self.args, "pruning_T_end", self.T_end) or self.T_end)
        self.final_iter = int(self.pruning_T_end / self.delta_T)
        self.ini_iter = int(getattr(self.args, "granet_init_step", 0) / self.delta_T)
        self.total_prune_iter = max(1, self.final_iter - self.ini_iter)

        # mask buffers
        self.backward_masks: List[Optional[Mask]] = []
        self.record_mask: List[Mask] = []

        # optionally early-stop flag
        self.early_stop_signal = torch.zeros(len(self.W)) if getattr(self.args, "early_stop", False) else None

        # hook objects for dense-grad collection
        self.backward_hook_objects: List[Optional[IndexMaskHook]] = []

        # wrap optimizer.step
        _wrap_optimizer_step(self, optimizer)

        # restore or init
        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()
        else:
            # initial per-layer target sparsities
            self.S: List[float] = []
            for _ in self.W:
                if getattr(self.args, "EM_S", False):
                    self.S.append(1.0 - self.dense_allocation - 0.05)
                else:
                    self.S.append(1.0 - self.dense_allocation)

            if getattr(self.args, "init_mode", "") in ("swi", "kaiming"):
                self.reset_parameters()

            # history (optional)
            self.history_masks: Optional[List[torch.Tensor]] = None
            if getattr(self.args, "history_weights", False):
                self.history_masks = [w.detach().clone().cpu() for w in self.W]

            self.random_sparsify()

            if getattr(self.args, "new_history_weights", False):
                self.history_masks = [w.detach().clone().cpu() for w in self.W]

            self.step = 0
            self.dst_steps = 0

        # register backward hooks (mask grads + dense grad accumulation)
        for i, w in enumerate(self.W):
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                self.backward_masks.append(None) if len(self.backward_masks) < len(self.W) else None
                continue

            if getattr(w, "_has_rigl_backward_hook", False):
                raise RuntimeError("This parameter already has a DST backward hook registered.")

            hook = IndexMaskHook(i, self)
            w.register_hook(hook)
            setattr(w, "_has_rigl_backward_hook", True)

            self.backward_hook_objects.append(hook)

        if self.rank == 0:
            print(f"[DST] ini_iter={self.ini_iter}, final_iter={self.final_iter}, total_prune_iter={self.total_prune_iter}")

        if self.sparsity_distribution not in ("uniform", "non-uniform"):
            raise ValueError("sparsity_distribution must be 'uniform' or 'non-uniform'")

    # --------- state ---------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dense_allocation": self.dense_allocation,
            "global_sparsity": self.global_sparsity,
            "S": self.S,
            "N": self.N,
            "delta_T": self.delta_T,
            "alpha": self.alpha,
            "T_end": self.T_end,
            "ignore_linear_layers": self.ignore_linear_layers,
            "static_topo": self.static_topo,
            "sparsity_distribution": self.sparsity_distribution,
            "grad_accumulation_n": self.grad_accumulation_n,
            "step": self.step,
            "dst_steps": self.dst_steps,
            "backward_masks": self.backward_masks,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        # minimal, explicit restore (avoid accidental recursion / clobber)
        self.dense_allocation = float(sd["dense_allocation"])
        self.global_sparsity = float(sd.get("global_sparsity", 1.0 - self.dense_allocation))
        self.S = list(sd["S"])
        self.N = list(sd["N"])
        self.delta_T = int(sd["delta_T"])
        self.alpha = float(sd["alpha"])
        self.T_end = int(sd["T_end"])
        self.ignore_linear_layers = bool(sd["ignore_linear_layers"])
        self.static_topo = bool(sd["static_topo"])
        self.sparsity_distribution = str(sd["sparsity_distribution"])
        self.grad_accumulation_n = int(sd["grad_accumulation_n"])
        self.step = int(sd["step"])
        self.dst_steps = int(sd["dst_steps"])
        self.backward_masks = sd["backward_masks"]

    # --------- helpers ---------

    def _build_layer_meta(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """
        Build per-layer metadata from modules annotated by dst_util.get_W(..., annotate_modules=True).
        For attention q/k/v projections, record num_heads and per-head row span so that
        remove/regrow can run independently per head.
        """
        meta: List[Dict[str, Any]] = []
        for w in self.W:
            meta.append({
                "name": "",
                "kind": "",
                "per_head": False,
                "num_heads": 1,
                "head_rows": int(w.shape[0]),
            })

        modules_by_name = dict(model.named_modules())
        model_heads = int(getattr(getattr(model, "config", None), "num_attention_heads", 0) or 0)

        for full_name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if not hasattr(mod, "LAYER_INDEX"):
                continue

            idx = int(getattr(mod, "LAYER_INDEX"))
            if idx < 0 or idx >= len(meta):
                continue

            kind = full_name.split(".")[-1]
            row_dim = int(mod.weight.shape[0])

            meta[idx]["name"] = full_name
            meta[idx]["kind"] = kind
            meta[idx]["head_rows"] = row_dim

            if kind not in ("q_proj", "k_proj", "v_proj"):
                continue

            parent_name = ".".join(full_name.split(".")[:-1])
            parent = modules_by_name.get(parent_name, None)

            n_heads = 0
            if parent is not None:
                n_heads = int(getattr(parent, "num_heads", 0) or 0)
                if n_heads <= 0:
                    n_heads = int(getattr(parent, "num_key_value_heads", 0) or 0)
            if n_heads <= 0:
                n_heads = model_heads

            if n_heads > 0 and row_dim % n_heads == 0:
                meta[idx]["per_head"] = True
                meta[idx]["num_heads"] = n_heads
                meta[idx]["head_rows"] = row_dim // n_heads

        return meta

    def _is_attention_qkv_layer(self, layer_idx: int) -> bool:
        info = self.layer_meta[layer_idx]
        return bool(info.get("per_head", False))

    def _head_row_slices(self, layer_idx: int, n_rows: int) -> List[slice]:
        """
        Return row slices for per-head processing.
        Non q/k/v layers return a single full-matrix slice.
        """
        info = self.layer_meta[layer_idx]
        if not bool(info.get("per_head", False)):
            return [slice(0, n_rows)]

        n_heads = int(info.get("num_heads", 1))
        head_rows = int(info.get("head_rows", n_rows))
        if n_heads <= 0 or head_rows <= 0 or n_heads * head_rows != n_rows:
            return [slice(0, n_rows)]

        return [slice(h * head_rows, (h + 1) * head_rows) for h in range(n_heads)]

    def _paired_head_slices(
        self,
        layer_a: int,
        layer_b: int,
        rows_a: int,
        rows_b: int,
    ) -> Optional[List[Tuple[slice, slice]]]:
        """
        Return per-head paired row slices for q/k chain-removal when both layers
        are attention q/k/v projections with matching head count.
        """
        info_a = self.layer_meta[layer_a]
        info_b = self.layer_meta[layer_b]
        if not (bool(info_a.get("per_head", False)) and bool(info_b.get("per_head", False))):
            return None

        n_heads_a = int(info_a.get("num_heads", 1))
        n_heads_b = int(info_b.get("num_heads", 1))
        head_rows_a = int(info_a.get("head_rows", rows_a))
        head_rows_b = int(info_b.get("head_rows", rows_b))
        if n_heads_a <= 0 or n_heads_a != n_heads_b:
            return None
        if n_heads_a * head_rows_a != rows_a or n_heads_b * head_rows_b != rows_b:
            return None

        out: List[Tuple[slice, slice]] = []
        for h in range(n_heads_a):
            sa = slice(h * head_rows_a, (h + 1) * head_rows_a)
            sb = slice(h * head_rows_b, (h + 1) * head_rows_b)
            out.append((sa, sb))
        return out

    @staticmethod
    def _topk_keep_mask(score: torch.Tensor, n_keep: int) -> torch.Tensor:
        flat = score.reshape(-1)
        if n_keep <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        if n_keep >= flat.numel():
            return torch.ones_like(score, dtype=torch.bool)

        _, keep_idx = torch.topk(flat, k=n_keep, largest=True, sorted=False)
        out = torch.zeros_like(flat, dtype=torch.bool)
        out[keep_idx] = True
        return out.view_as(score)

    @staticmethod
    def _soft_sample_keep_mask(score: torch.Tensor, n_keep: int, temperature: float) -> torch.Tensor:
        flat = (score.reshape(-1).clamp_min(1e-12)) ** temperature
        n_keep = min(max(0, int(n_keep)), int(flat.numel()))
        if n_keep <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        if n_keep >= flat.numel():
            return torch.ones_like(score, dtype=torch.bool)

        s = flat.sum()
        if not torch.isfinite(s).item() or float(s.item()) <= 0.0:
            keep_idx = torch.randperm(flat.numel(), device=flat.device)[:n_keep]
        else:
            probs = flat / s
            keep_idx = torch.multinomial(probs, n_keep, replacement=False)

        out = torch.zeros_like(flat, dtype=torch.bool)
        out[keep_idx] = True
        return out.view_as(score)

    def should_accumulate_dense_grad(self) -> bool:
        if self.step >= self.T_end:
            return False
        steps_til_next = self.delta_T - (self.step % self.delta_T)
        return steps_til_next <= self.grad_accumulation_n

    def cosine_annealing(self) -> float:
        return float(self.alpha / 2.0 * (1.0 + np.cos((self.step * np.pi) / max(1, self.T_end))))

    def __str__(self) -> str:
        # Prefer mask-based stats (cheap + consistent) rather than scanning weights for zeros.
        total_params = 0
        total_nonzero = 0
        parts = []

        for n, s, mask in zip(self.N, self.S, self.backward_masks):
            if mask is None or s <= 0:
                parts.append(f"{n}/{n} (100.00%)")
                total_params += n
                total_nonzero += n
                continue
            nnz = int(mask.sum().item())
            total_params += n
            total_nonzero += nnz
            parts.append(f"{nnz}/{n} ({(100.0*nnz/n):.2f}%)")

        self.global_sparsity = 1.0 - (total_nonzero / max(1, total_params))
        return (
            "DSTScheduler("
            f"layers={len(self.N)}, "
            f"nonzero={parts}, "
            f"global_density={(100.0*total_nonzero/max(1,total_params)):.2f}%, "
            f"step={getattr(self, 'step', 0)}, dst_steps={getattr(self, 'dst_steps', 0)}, "
            f"target_sparsity={getattr(self.args, 'sparsity', None)})"
        )

    # --------- mask application ---------

    @torch.no_grad()
    def apply_mask_to_weights(self) -> None:
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0 or mask is None:
                continue
            w.mul_(mask)

    @torch.no_grad()
    def apply_mask_to_gradients(self) -> None:
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0 or mask is None or w.grad is None:
                continue
            w.grad.mul_(mask)

    @torch.no_grad()
    def reset_momentum(self) -> None:
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0 or mask is None:
                continue

            param_state = self.optimizer.state.get(w, {})
            base_param_state = None
            if getattr(self.args, "ssam", False):
                base_param_state = self.optimizer.base_optimizer.state.get(w, {})  # type: ignore[attr-defined]

            for key in ("exp_avg", "exp_avg_sq", "prev_grad", "prev_u", "e_w"):
                if key in param_state:
                    param_state[key].mul_(mask)
                if base_param_state is not None and key in base_param_state:
                    base_param_state[key].mul_(mask)

    # --------- init sparsity ---------

    @torch.no_grad()
    def random_sparsify(self) -> None:
        self.backward_masks = []
        self.record_mask = []

        for l, w in enumerate(self.W):
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            # Rank0 creates mask, then broadcasts (DDP-safe & deterministic)
            if self.is_dist and self.rank != 0:
                mask = torch.empty_like(w, dtype=torch.bool)
                dist.broadcast(mask, 0)
            else:
                if getattr(self.args, "WS", False):
                    mask = create_ws_sparse_scheduler(self.S[l], w, self.args).contiguous().bool()
                elif getattr(self.args, "BRF", False):
                    mask = create_brf_sparse_scheduler(self.S[l], w, self.args).contiguous().bool()
                else:
                    n = w.numel()
                    n_zero = int(self.S[l] * n)
                    n_keep = max(0, n - n_zero)
                    flat = torch.zeros(n, device=w.device, dtype=torch.bool)
                    if n_keep > 0:
                        keep_idx = torch.randperm(n, device=w.device)[:n_keep]
                        flat[keep_idx] = True
                    mask = flat.view_as(w)

                if self.is_dist:
                    dist.broadcast(mask, 0)

            w.mul_(mask)
            self.backward_masks.append(mask)
            if getattr(self.args, "itop", False):
                self.record_mask.append(mask.clone())

    # --------- main scheduling ---------

    def __call__(self) -> bool:
        """
        Return value convention (to match your training loop):
          True  -> optimizer.step() should run
          False -> skip optimizer.step() this iteration (because _dst_step performed)
        """
        self.step += 1

        if self.static_topo:
            return True

        if getattr(self.args, "early_stop", False) and self.early_stop_signal is not None:
            if int(self.early_stop_signal.sum().item()) == len(self.W):
                return True

        if (self.step % self.delta_T) == 0 and self.step <= self.T_end:
            self._dst_step()
            self.dst_steps += 1
            if self.rank == 0:
                print(self)
            return False

        if (self.step % self.delta_T) == 0 and self.rank == 0:
            print(self)

        return True

    # -----------------------------
    # Pruning (GraNet/GMP)
    # -----------------------------

    @torch.no_grad()
    def _current_prune_rate(self) -> float:
        curr_prune_iter = int((self.step - self.ini_iter) / self.delta_T)

        sched = getattr(self.args, "pruning_scheduler", "linear")
        s0 = float(getattr(self.args, "granet_init_sparsity", 0.0))
        s1 = float(getattr(self.args, "sparsity", 0.0))

        if sched == "linear":
            return (s1 - s0) * curr_prune_iter / self.total_prune_iter + s0

        if sched == "granet":
            prune_decay = (1.0 - (curr_prune_iter - self.ini_iter) / self.total_prune_iter) ** 3
            return s0 + (s1 - s0) * (1.0 - prune_decay)

        if sched == "s_shape":
            mid = self.total_prune_iter / 2.0
            k = float(getattr(self.args, "k", 6.0)) / max(1e-9, mid)
            # normalize sigmoid to map [0, total] -> [0,1]
            a0 = 1.0 / (1.0 + np.exp(-k * (-mid)))
            a1 = 1.0 / (1.0 + np.exp(-k * (self.total_prune_iter - mid)))
            scale = 1.0 / max(1e-9, (a1 - a0))
            x = 1.0 / (1.0 + np.exp(-k * (curr_prune_iter - mid)))
            shaped = ((x - 0.5) * scale + 0.5)
            return (s1 - s0) * shaped + s0

        raise NotImplementedError(f"Unknown pruning_scheduler: {sched}")

    @torch.no_grad()
    def uniform_pruning(self) -> None:
        curr_prune_rate = self._current_prune_rate()

        if self.rank == 0:
            curr_prune_iter = int((self.step - self.ini_iter) / self.delta_T)
            print("*" * 54)
            print(f"Pruning progress: {curr_prune_iter - self.ini_iter} / {self.total_prune_iter}")
            print("*" * 54)

        for l, w in enumerate(self.W):
            if self.S[l] <= 0 or self.backward_masks[l] is None:
                continue

            # choose weights to score
            if getattr(self.args, "history_weights", False) and not getattr(self.args, "new_history_weights", False):
                weight = self.history_masks[l].to(w.device)  # type: ignore[index]
            else:
                weight = w.detach()
                if self.is_dist:
                    tmp = weight.clone()
                    dist.all_reduce(tmp)
                    weight = tmp / self.world_size

            score = self._score_for_pruning(l, weight)

            flat = score.flatten()
            keep = max(1, int(flat.numel() * (1.0 - curr_prune_rate)))
            topk_vals, topk_idx = torch.topk(flat, k=keep, largest=True, sorted=False)

            new_mask = torch.zeros_like(flat, dtype=torch.bool)
            new_mask[topk_idx] = True
            new_mask = new_mask.view_as(score)

            if self.is_dist:
                dist.broadcast(new_mask, 0)

            self.backward_masks[l] = new_mask
            self.S[l] = 1.0 - float(new_mask.sum().item()) / float(self.N[l])

    @torch.no_grad()
    def non_uniform_pruning(self) -> None:
        curr_prune_rate = self._current_prune_rate()

        if self.rank == 0:
            curr_prune_iter = int((self.step - self.ini_iter) / self.delta_T)
            print("*" * 54)
            print(f"Pruning progress: {curr_prune_iter - self.ini_iter} / {self.total_prune_iter}")
            print("*" * 54)

        scores_per_layer: List[torch.Tensor] = []
        layer_ids: List[int] = []

        for l, w in enumerate(self.W):
            if self.S[l] <= 0 or self.backward_masks[l] is None:
                continue

            if getattr(self.args, "history_weights", False):
                weight = self.history_masks[l].to(w.device)  # type: ignore[index]
            else:
                weight = w.detach()
                if self.is_dist:
                    tmp = weight.clone()
                    dist.all_reduce(tmp)
                    weight = tmp / self.world_size

            scores_per_layer.append(self._score_for_pruning(l, weight))
            layer_ids.append(l)

        all_scores = torch.cat([x.flatten() for x in scores_per_layer])
        keep = max(1, int(all_scores.numel() * (1.0 - curr_prune_rate)))
        _, topk_idx = torch.topk(all_scores, k=keep, largest=True, sorted=False)
        global_mask = torch.zeros_like(all_scores, dtype=torch.bool)
        global_mask[topk_idx] = True

        # scatter back to layers
        offset = 0
        total_size = 0
        total_keep = 0

        for scores, l in zip(scores_per_layer, layer_ids):
            numel = scores.numel()
            layer_keep = global_mask[offset: offset + numel].view_as(scores)
            offset += numel

            if self.is_dist:
                dist.broadcast(layer_keep, 0)

            self.backward_masks[l] = layer_keep
            total_size += self.N[l]
            total_keep += int(layer_keep.sum().item())
            self.S[l] = 1.0 - float(layer_keep.sum().item()) / float(self.N[l])

        if self.rank == 0:
            print("Total model parameters:", total_size)
            print("Density after pruning:", total_keep / max(1, total_size))

    @torch.no_grad()
    def _score_for_pruning(self, layer_idx: int, weight: torch.Tensor) -> torch.Tensor:
        method = getattr(self.args, "pruning_method", "weight_magnitude")

        if method == "weight_magnitude":
            return weight.abs()

        if method == "ri":
            eps = 1e-5
            wabs = weight.abs()
            return wabs / (wabs.sum(dim=0) + eps) + wabs / (wabs.sum(dim=1).view(-1, 1) + eps)

        if method == "MEST":
            hook = self.backward_hook_objects[layer_idx]
            if hook is None or hook.dense_grad is None:
                return weight.abs()
            score_grow = hook.dense_grad.abs()
            if self.is_dist:
                tmp = score_grow.clone()
                dist.all_reduce(tmp)
                score_grow = tmp / self.world_size
            return weight.abs() + float(getattr(self.args, "factor", 0.01)) * score_grow.abs()

        raise NotImplementedError(f"Unknown pruning_method: {method}")

    # -----------------------------
    # DST step: prune / remove / regrow
    # -----------------------------

    @torch.no_grad()
    def _dst_step(self) -> None:
        if getattr(self.args, "EM_S", False) and getattr(self.args, "adaptive_zeta", False):
            raise NotImplementedError("EM_S and adaptive_zeta cannot be used together.")

        # save history (active weights only)
        if getattr(self.args, "history_weights", False) and self.history_masks is not None:
            for l, w in enumerate(self.W):
                w_avg = w.detach()
                if self.is_dist:
                    tmp = w_avg.clone()
                    dist.all_reduce(tmp)
                    w_avg = tmp / self.world_size
                mask = self.backward_masks[l]
                if mask is not None:
                    self.history_masks[l][mask] = w_avg[mask].cpu()

        # gradual pruning stage
        if self.step <= self.pruning_T_end and (getattr(self.args, "granet", False) or getattr(self.args, "gmp", False)):
            if self.sparsity_distribution == "non-uniform":
                self.non_uniform_pruning()
            else:
                self.uniform_pruning()

            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients()

        # GMP only: optionally chain removal and stop
        if getattr(self.args, "gmp", False):
            if getattr(self.args, "chain_removal", False):
                self.chain_removal()
                self.reset_momentum()
                self.apply_mask_to_weights()
                self.apply_mask_to_gradients()
            return

        # regular DST: remove + optional chain removal + regrow
        self.link_removal()
        if getattr(self.args, "chain_removal", False):
            self.chain_removal()
        self.link_regrowth()

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()

        torch.cuda.empty_cache()

    @torch.no_grad()
    def link_removal(self) -> None:
        for l, w in enumerate(self.W):
            if self.S[l] <= 0 or self.backward_masks[l] is None:
                continue

            if getattr(self.args, "EM_S", False):
                drop_fraction = (1.0 - self.S[l] - self.dense_allocation) / max(1e-9, (1.0 - self.S[l]))
            elif getattr(self.args, "adaptive_zeta", False):
                drop_fraction = self.cosine_annealing()
            else:
                drop_fraction = float(self.alpha)

            current_mask = self.backward_masks[l]

            # Score weights (global average if dist)
            score_abs = w.detach().abs()
            if self.is_dist:
                tmp = score_abs.clone()
                dist.all_reduce(tmp)
                score_abs = tmp / self.world_size

            method = str(getattr(self.args, "remove_method", "weight_magnitude"))

            score_grow = None
            if method == "MEST":
                hook = self.backward_hook_objects[l]
                if hook is not None and hook.dense_grad is not None:
                    score_grow = hook.dense_grad.abs()
                    if self.is_dist:
                        tmp = score_grow.clone()
                        dist.all_reduce(tmp)
                        score_grow = tmp / self.world_size

            if method.endswith("_soft"):
                T0 = float(getattr(self.args, "start_T", 1.0))
                T1 = float(getattr(self.args, "end_T", 1.0))
                T = T0 + self.step * ((T1 - T0) / max(1, self.T_end))
            else:
                T = 1.0

            new_mask = torch.zeros_like(current_mask, dtype=torch.bool)
            for row_slice in self._head_row_slices(l, current_mask.shape[0]):
                head_mask = current_mask[row_slice, :]
                head_score_abs = score_abs[row_slice, :]

                head_n_ones = int(head_mask.sum().item())
                head_n_prune = int(head_n_ones * drop_fraction)
                head_n_keep = max(0, head_n_ones - head_n_prune)

                if method in ("weight_magnitude", "ri", "MEST"):
                    if method == "ri":
                        eps = 1e-5
                        head_score = (
                            head_score_abs / (head_score_abs.sum(dim=0) + eps)
                            + head_score_abs / (head_score_abs.sum(dim=1).view(-1, 1) + eps)
                        )
                    elif method == "MEST" and score_grow is not None:
                        head_score = head_score_abs + float(getattr(self.args, "factor", 0.01)) * (
                            score_grow[row_slice, :].abs() * head_mask
                        )
                    else:
                        head_score = head_score_abs

                    head_new = self._topk_keep_mask(head_score, head_n_keep)

                elif method.endswith("_soft"):
                    if method.startswith("weight_magnitude"):
                        head_score_drop = head_score_abs
                    elif method.startswith("ri"):
                        eps = 1e-5
                        head_score_drop = (
                            head_score_abs / (head_score_abs.sum(dim=0) + eps)
                            + head_score_abs / (head_score_abs.sum(dim=1).view(-1, 1) + eps)
                        )
                    else:
                        raise NotImplementedError(f"Unknown soft remove_method: {method}")

                    # keep behavior consistent with old code: soft branch keeps at least one link.
                    head_new = self._soft_sample_keep_mask(head_score_drop, max(1, head_n_keep), T)

                else:
                    raise NotImplementedError(f"Unknown remove_method: {method}")

                new_mask[row_slice, :] = head_new

            self.backward_masks[l] = new_mask.to(device=w.device, dtype=torch.bool)

    @torch.no_grad()
    def chain_removal(self) -> None:
        for a, b in self.qk_chain_list:
            ma, mb = self.backward_masks[a], self.backward_masks[b]
            if ma is None or mb is None:
                continue

            paired_slices = self._paired_head_slices(a, b, ma.shape[0], mb.shape[0])
            if paired_slices is None:
                self.backward_masks[a], self.backward_masks[b] = qk_chain_removal(ma, mb)
                continue

            ma_new = ma.clone()
            mb_new = mb.clone()
            for sa, sb in paired_slices:
                qa, kb = qk_chain_removal(ma[sa, :], mb[sb, :])
                ma_new[sa, :] = qa
                mb_new[sb, :] = kb
            self.backward_masks[a], self.backward_masks[b] = ma_new, mb_new

        for a, b in self.chain_list:
            ma, mb = self.backward_masks[a], self.backward_masks[b]
            if ma is None or mb is None:
                continue
            self.backward_masks[a], self.backward_masks[b] = chain_removal(ma, mb)

    @torch.no_grad()
    def link_regrowth(self) -> None:
        for l, w in enumerate(self.W):
            if self.S[l] <= 0 or self.backward_masks[l] is None:
                continue

            # how many to regrow
            if getattr(self.args, "EM_S", False):
                if self.step <= self.T_end * 0.6:
                    self.S[l] = 1.0 - self.dense_allocation - 0.05
                    n_regrow = int(0.05 * self.N[l])
                elif self.step < (self.T_end - self.delta_T):
                    self.S[l] = 1.0 - self.dense_allocation - 0.025
                    n_regrow = int(0.025 * self.N[l])
                else:
                    self.S[l] = 1.0 - self.dense_allocation
                    n_regrow = 0
            else:
                target_nnz = int((1.0 - self.S[l]) * self.N[l])
                current_nnz = int(self.backward_masks[l].sum().item())
                n_regrow = target_nnz - current_nnz

            if n_regrow <= 0 and (not self._is_attention_qkv_layer(l)):
                continue

            current_mask = self.backward_masks[l].clone()

            # scores for regrowth (only on zeros)
            scores = self._scores_for_regrowth(l, current_mask, w)
            scores = scores * (~current_mask)  # only consider zero positions

            # select top-k positions
            new_links = torch.zeros_like(current_mask, dtype=torch.bool)
            if self._is_attention_qkv_layer(l):
                # q/k/v regrow independently per head
                for row_slice in self._head_row_slices(l, current_mask.shape[0]):
                    head_mask = current_mask[row_slice, :]
                    head_scores = scores[row_slice, :]

                    head_target_nnz = int((1.0 - self.S[l]) * head_mask.numel())
                    head_current_nnz = int(head_mask.sum().item())
                    head_regrow = head_target_nnz - head_current_nnz
                    if head_regrow <= 0:
                        continue

                    head_flat = head_scores.reshape(-1)
                    head_candidates = int((~head_mask).sum().item())
                    if head_candidates <= 0:
                        continue

                    k = min(max(1, int(head_regrow)), max(1, head_candidates))
                    if k >= head_flat.numel():
                        head_grow_flat = head_flat > -1  # all True
                    else:
                        _, idx = torch.topk(head_flat, k=k, largest=True, sorted=False)
                        head_grow_flat = torch.zeros_like(head_flat, dtype=torch.bool)
                        head_grow_flat[idx] = True
                    new_links[row_slice, :] = head_grow_flat.view_as(head_mask)
            else:
                flat = scores.view(-1)
                num_candidates = int((~current_mask).sum().item())
                k = min(max(1, int(n_regrow)), max(1, num_candidates))

                if k >= flat.numel():
                    grow_flat = flat > -1  # all True
                else:
                    _, idx = torch.topk(flat, k=k, largest=True, sorted=False)
                    grow_flat = torch.zeros_like(flat, dtype=torch.bool)
                    grow_flat[idx] = True
                new_links = grow_flat.view_as(current_mask)

            self.backward_masks[l] = (current_mask | new_links)

            if self.is_dist:
                dist.broadcast(self.backward_masks[l], 0)

            if getattr(self.args, "itop", False) and l < len(self.record_mask):
                self.record_mask[l] = (self.record_mask[l] | self.backward_masks[l])
                if self.rank == 0:
                    print("ITOP rate:", (self.record_mask[l].sum().item() / self.N[l]))

    @torch.no_grad()
    def _scores_for_regrowth(self, layer_idx: int, current_mask: Mask, w: torch.Tensor) -> torch.Tensor:
        method = str(getattr(self.args, "regrow_method", "random")).lower()

        if method == "random":
            return torch.rand_like(w)

        if method == "gradient":
            hook = self.backward_hook_objects[layer_idx]
            if hook is None or hook.dense_grad is None:
                # fallback
                return torch.zeros_like(w)
            return hook.dense_grad.abs()

        if "ch" in method:
            # Expect patterns like "CH2_L3n", "CH3_L3p", etc. (case-insensitive)
            ch_method = method.split("_")[0].upper()
            if self._is_attention_qkv_layer(layer_idx):
                scores = torch.zeros_like(w)
                for row_slice in self._head_row_slices(layer_idx, current_mask.shape[0]):
                    head_scores = self._scores_for_regrowth_ch(
                        method=method,
                        ch_method=ch_method,
                        current_mask=current_mask[row_slice, :],
                        w=w[row_slice, :],
                    )
                    scores[row_slice, :] = head_scores
                return scores

            return self._scores_for_regrowth_ch(
                method=method,
                ch_method=ch_method,
                current_mask=current_mask,
                w=w,
            )

        raise NotImplementedError(f"Unknown regrow_method: {method}")

    @torch.no_grad()
    def _scores_for_regrowth_ch(
        self,
        *,
        method: str,
        ch_method: str,
        current_mask: Mask,
        w: torch.Tensor,
    ) -> torch.Tensor:
        if "l3n" in method:
            dt = current_mask.float()
            td = dt.transpose(1, 0)

            dd2 = dt @ td
            tt2 = td @ dt

            bdd2 = dd2 != 0
            btt2 = tt2 != 0

            elcl_dt = (dt.sum(dim=1) - dd2) * bdd2
            elcl_td = (td.sum(dim=1) - tt2) * btt2

            elcl_dt = elcl_dt.clamp_min(1) - 1
            elcl_td = elcl_td.clamp_min(1) - 1

            if ch_method == "CH2":
                elcl_dt = (dd2 + bdd2) / (elcl_dt + 1)
                elcl_td = (tt2 + btt2) / (elcl_td + 1)
            elif ch_method == "CH3":
                elcl_dt = bdd2 / (elcl_dt + 1)
                elcl_td = btt2 / (elcl_td + 1)
            elif ch_method == "CH3.1":
                elcl_dt = (dd2 + bdd2) / ((elcl_dt + 1) ** (1 + (elcl_dt / (1 + elcl_dt))))
                elcl_td = (tt2 + btt2) / ((elcl_td + 1) ** (1 + (elcl_td / (1 + elcl_td))))
            else:
                raise NotImplementedError(f"Unsupported CH method: {ch_method}")

            elcl_dt = elcl_dt @ dt
            elcl_td = elcl_td @ td
            return (elcl_dt + elcl_td.transpose(1, 0)).to(w.device)

        if "l3p" in method:
            if CH_scores is None:
                raise ImportError("CH_scores is required for *_L3p regrow_method but could not be imported.")

            xb = current_mask.detach().cpu().numpy()
            x = transform_bi_to_mo(xb)
            A = csr_matrix(x)
            ir, jc = A.indices, A.indptr

            if ch_method == "CH2":
                sc = CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [2], 1)
            elif ch_method == "CH3":
                sc = CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [3], 1)
            elif ch_method == "CH3.1":
                sc = CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [5], 1)
            else:
                raise NotImplementedError(f"Unsupported CH method: {ch_method}")

            scores = torch.tensor(np.array(sc), device=w.device).view(*x.shape)
            scores = scores[: xb.shape[0], xb.shape[0] :]
            return scores

        raise NotImplementedError(f"Unsupported CH regrow_method variant: {method}")

    # -----------------------------
    # Optional init reset
    # -----------------------------

    @torch.no_grad()
    def reset_parameters(self) -> None:
        mode = str(getattr(self.args, "init_mode", "")).lower()
        for l, w in enumerate(self.W):
            if mode == "swi":
                # keep your original logic but make it numerically safe
                eff_nnz = max(1.0, (1.0 - self.S[l]) * self.N[l])
                stdv = math.sqrt(2.0 / max(1e-9, (eff_nnz / w.shape[1])))
            elif mode == "kaiming":
                stdv = math.sqrt(2.0 / w.shape[1])
            else:
                raise NotImplementedError(f"Unknown init_mode: {mode}")
            w.copy_(torch.randn_like(w) * stdv)


# -----------------------------
# Utility
# -----------------------------

def transform_bi_to_mo(xb: np.ndarray) -> np.ndarray:
    """
    Convert bipartite adjacency (n_left, n_right) to monopartite block matrix.
    """
    n0, n1 = xb.shape
    x = np.zeros((n0 + n1, n0 + n1), dtype=xb.dtype)
    x[:n0, n0:] = xb
    x[n0:, :n0] = xb.T
    return x
