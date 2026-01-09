from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


TARGET_MODULE_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


@dataclass(frozen=True)
class LayerChains:
    """Indices describing relationships between collected projection layers."""
    chain: List[Tuple[int, int]]
    qk_chain: List[Tuple[int, int]]


def _is_target_linear(name: str, module: nn.Module) -> bool:
    # Restrict to Linear modules with canonical names used in LLaMA-style blocks.
    return isinstance(module, nn.Linear) and name.split(".")[-1] in TARGET_MODULE_NAMES


def get_projection_weights_and_chains(
    model: nn.Module,
    *,
    annotate_modules: bool = False,
    index_attr: str = "LAYER_INDEX",
) -> Tuple[List[torch.Tensor], LayerChains]:
    """
    Collect projection weights (q/k/v/o + mlp gate/up/down) in a deterministic order
    and build chain indices for downstream algorithms.

    Ordering:
      - Deterministic traversal via model.named_modules()
      - Within a block: attention k,v,q,o then MLP gate,up,down (matching your original)

    Chains:
      - qk_chain: (k, q) for attention, (gate, up) for MLP
      - chain: (v, o) for attention; (gate, down), (up, down) for MLP

    Args:
        model: PyTorch module (e.g., HF LlamaForCausalLM or your custom LlamaForCausalLM).
        annotate_modules: If True, set `index_attr` on each collected Linear module.
        index_attr: Attribute name used when annotate_modules=True.

    Returns:
        weights: list of parameter tensors (module.weight) in collected order
        chains: LayerChains(chain=..., qk_chain=...)
    """
    # 1) Collect target Linear modules + remember their full names for indexing.
    # We skip lm_head by name prefix, same as your original.
    linear_modules: List[Tuple[str, nn.Linear]] = []
    for full_name, mod in model.named_modules():
        if full_name.endswith("lm_head") or full_name.startswith("lm_head."):
            continue
        if _is_target_linear(full_name, mod):
            linear_modules.append((full_name, mod))

    # 2) Enforce a stable, intended order:
    # group by parent path (…self_attn / …mlp) then apply local ordering inside each group.
    def parent_path(n: str) -> str:
        return ".".join(n.split(".")[:-1])

    local_order = {
        "k_proj": 0, "v_proj": 1, "q_proj": 2, "o_proj": 3,
        "gate_proj": 0, "up_proj": 1, "down_proj": 2,
    }

    # Sort key:
    #   (parent_path, module_kind_rank, local_order)
    # module_kind_rank makes self_attn come before mlp within same layer if both appear.
    def sort_key(item: Tuple[str, nn.Linear]) -> Tuple[str, int, int]:
        full_name, _ = item
        last = full_name.split(".")[-1]
        parent = parent_path(full_name)
        kind_rank = 0 if parent.endswith("self_attn") else (1 if parent.endswith("mlp") else 2)
        return (parent, kind_rank, local_order.get(last, 999))

    linear_modules.sort(key=sort_key)

    # 3) Build weights and index map.
    weights: List[torch.Tensor] = []
    name_to_index: dict[str, int] = {}
    for idx, (full_name, lin) in enumerate(linear_modules):
        weights.append(lin.weight)
        name_to_index[full_name] = idx
        if annotate_modules:
            setattr(lin, index_attr, idx)

    # 4) Build chains by scanning parent blocks.
    # We create parent -> {subname: full_name} maps for self_attn and mlp.
    chain: List[Tuple[int, int]] = []
    qk_chain: List[Tuple[int, int]] = []

    # Index by parent module path
    by_parent: dict[str, dict[str, str]] = {}
    for full_name, _ in linear_modules:
        parent = parent_path(full_name)  # e.g. "...layers.0.self_attn"
        sub = full_name.split(".")[-1]   # e.g. "q_proj"
        by_parent.setdefault(parent, {})[sub] = full_name

    for parent, subs in by_parent.items():
        if parent.endswith("self_attn"):
            # Need k,q and v,o
            if all(k in subs for k in ("k_proj", "q_proj")):
                qk_chain.append((name_to_index[subs["k_proj"]], name_to_index[subs["q_proj"]]))
            if all(k in subs for k in ("v_proj", "o_proj")):
                chain.append((name_to_index[subs["v_proj"]], name_to_index[subs["o_proj"]]))

        elif parent.endswith("mlp"):
            # Need gate, up, down
            if all(k in subs for k in ("gate_proj", "up_proj")):
                qk_chain.append((name_to_index[subs["gate_proj"]], name_to_index[subs["up_proj"]]))
            if all(k in subs for k in ("gate_proj", "down_proj")):
                chain.append((name_to_index[subs["gate_proj"]], name_to_index[subs["down_proj"]]))
            if all(k in subs for k in ("up_proj", "down_proj")):
                chain.append((name_to_index[subs["up_proj"]], name_to_index[subs["down_proj"]]))

    return weights, LayerChains(chain=chain, qk_chain=qk_chain)


def get_W(model: nn.Module, args=None):
    """
    Backward-compatible wrapper: returns (W_list, chain_list, qk_chain_list)
    to match your original function signature.
    """
    weights, chains = get_projection_weights_and_chains(model, annotate_modules=True)
    return weights, chains.chain, chains.qk_chain
