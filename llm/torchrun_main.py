#!/usr/bin/env python3
"""
Training entrypoint (DDP-friendly) for LLaMA-style CausalLM pretraining + optional DST scheduler.

Key properties:
- Works with torchrun (DDP) or --single_gpu
- Clean logging + rank0-only side effects
- Checkpoint save/restore for model/optimizer/scheduler/training state
- No hidden global args usage inside helpers
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import random
import argparse
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed

from tqdm import tqdm
from loguru import logger

import wandb

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

# Your local utils (must exist in repo)
from utils import *  # noqa: F403,F401

from dst_scheduler import DSTScheduler

transformers.logging.set_verbosity_error()


# -------------------------
# Small infra utilities
# -------------------------

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def rank0_only() -> bool:
    return get_rank() == 0


def safe_barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualWriter:
    """Write stdout to both console and a file (rank0-only recommended)."""
    def __init__(self, file_obj):
        self.file = file_obj
        self.console = sys.__stdout__

    def write(self, message: str) -> None:
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.console.flush()
        if not self.file.closed:
            self.file.flush()


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback) -> None:
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Uncaught Exception:\n{tb_str}")
    logger.error(f"Uncaught Exception:\n{tb_str}")


def setup_logging_to_file(enabled: bool, log_dir: str, run_tag: str) -> Optional[Any]:
    """Rank0-only: redirect stdout and loguru to file."""
    if not enabled or not rank0_only():
        return None

    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(log_dir, f"{run_tag}_{now}")

    txt_path = f"{base}.txt"
    log_path = f"{base}.log"

    f = open(txt_path, "w", encoding="utf-8")
    sys.stdout = DualWriter(f)

    logger.remove()
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
    logger.add(log_path, format="{time} {level} {message}", level="INFO", rotation="10 MB")

    sys.excepthook = log_uncaught_exceptions
    return f


def build_run_name(args: argparse.Namespace) -> str:
    """
    Keep user's --run_name as a prefix, then append a stable suffix that encodes key hyperparams.
    """
    suffix = f"step_{args.num_training_steps}_lr_{args.lr}_ui_{args.update_interval}_s_{args.sparsity}_z_{args.zeta}"

    if args.WS:
        suffix += f"_WS_{args.ws_beta}"
    if args.BRF:
        suffix += f"_BRF_{args.brf_r}_{args.degree_dist}"

    # encode remove_method temperatures without mutating args.remove_method
    rm = args.remove_method
    if "soft" in rm:
        suffix += f"_{rm}_startT_{args.start_T}_endT_{args.end_T}"
    else:
        suffix += f"_rm_{rm}"

    suffix += f"_rg_{args.regrow_method}_cr_{args.chain_removal}"

    if args.gmp:
        suffix += f"_gmp_{args.granet_init_sparsity}_{args.sparsity_distribution}"
    elif args.granet:
        suffix += f"_granet_{args.granet_init_sparsity}_{args.sparsity_distribution}_{args.pruning_method}_{args.pruning_scheduler}_{args.pruning_T_end}"
        suffix += f"_curvature_{args.k}"

    if args.EM_S:
        suffix += "_EM_S"
    elif args.adaptive_zeta:
        suffix += "_az"

    if args.history_weights:
        suffix += "_new_his" if args.new_history_weights else "_his"

    # Respect user prefix
    prefix = args.run_name.strip()
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


# -------------------------
# CLI
# -------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--run_name", type=str, default="", help="Run name prefix; auto suffix will be appended.")
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["openwebtext", "c4"])
    p.add_argument("--dataset_path", type=str, default="openwebtext", help="Used if dataset_name=openwebtext")
    p.add_argument("--use_hf_model", action="store_true", default=False)
    p.add_argument("--continue_from", type=str, default=None)
    p.add_argument("--tags", type=str, default=None)

    # batch
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--gradient_accumulation", type=int, default=None)
    p.add_argument("--total_batch_size", type=int, default=None)

    # training
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--num_training_steps", type=int, default=10_000)
    p.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None)
    p.add_argument("--eval_every", type=int, default=5_000)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--save_dir", type=str, default=None, help="Base output dir. If None, derived from run name.")
    p.add_argument("--only_save_last", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                   choices=["float32", "bfloat16", "bf16"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_clipping", type=float, default=0.0)
    p.add_argument("--scheduler", type=str, default="cosine_restarts", choices=["linear", "cosine", "cosine_restarts"])
    p.add_argument("--beta1", type=float, default=0.0, help="Momentum for SGD; beta1-like parameter.")

    # misc
    p.add_argument("--single_gpu", action="store_true", default=False, help="Disable torch.distributed, run single GPU.")
    p.add_argument("--no_log", action="store_true", default=False)
    p.add_argument("--no_decay", action="store_true", default=False)
    p.add_argument("--iterative_warmup_steps", type=int, default=0)
    p.add_argument("--log_to_file", action="store_true", default=False)
    p.add_argument("--log_dir", type=str, default="tmp", help="Folder for rank0 log files")

    # tokenizer
    p.add_argument("--tokenizer_name", type=str, default="t5-base",
                   help="Tokenizer name; OK because training from scratch (tokenizer choice is decoupled).")

    # DST parameters
    p.add_argument("--dst_scheduler", action="store_true", default=False)
    p.add_argument("--zeta", type=float, default=0.3)
    p.add_argument("--update_interval", type=int, default=200)
    p.add_argument("--sparsity", type=float, default=0.99)
    p.add_argument("--remove_method", type=str, default="weight_magnitude")
    p.add_argument("--regrow_method", type=str, default="random")
    p.add_argument("--init_mode", type=str, default="xavier")
    p.add_argument("--chain_removal", action="store_true", default=False)
    p.add_argument("--T_decay", type=str, default="no_decay")
    p.add_argument("--adaptive_zeta", action="store_true", default=False)

    # topology init
    p.add_argument("--WS", action="store_true", default=False)
    p.add_argument("--ws_beta", type=float, default=0.25)
    p.add_argument("--BRF", action="store_true", default=False)
    p.add_argument("--brf_r", type=float, default=0.25)
    p.add_argument("--degree_dist", type=str, default="uniform")
    

    p.add_argument("--itop", action="store_true", default=False)
    p.add_argument("--EM_S", action="store_true", default=False)
    p.add_argument("--factor", type=float, default=0.01)
    p.add_argument("--granet", action="store_true", default=False)
    p.add_argument("--granet_init_sparsity", type=float, default=0.9)
    p.add_argument("--granet_init_step", type=int, default=0)

    p.add_argument("--history_weights", action="store_true", default=False)
    p.add_argument("--new_history_weights", action="store_true", default=False)

    p.add_argument("--gmp", action="store_true", default=False)
    p.add_argument("--pruning_scheduler", type=str, default="none", choices=["none", "linear", "granet", "s_shape"])
    p.add_argument("--pruning_method", type=str, default="none", choices=["none", "ri", "weight_magnitude", "MEST"])
    p.add_argument("--sparsity_distribution", type=str, default="uniform")
    p.add_argument("--early_stop", action="store_true", default=False)
    p.add_argument("--pruning_T_end", type=float, default=None)

    p.add_argument("--start_T", type=float, default=1.0)
    p.add_argument("--end_T", type=float, default=1.0)
    p.add_argument("--k", type=float, default=6.0)


    args = p.parse_args(argv)
    args = args_utils.check_args_torchrun_main(args)

    # batch derivation
    if args.total_batch_size is not None and args.gradient_accumulation is None:
        ws = 1 if args.single_gpu else int(os.environ.get("WORLD_SIZE", "1"))
        assert args.total_batch_size % (args.batch_size * ws) == 0, \
            "total_batch_size must be divisible by batch_size * world_size"
        args.gradient_accumulation = args.total_batch_size // (args.batch_size * ws)

    if args.total_batch_size is None:
        ws = 1 if args.single_gpu else int(os.environ.get("WORLD_SIZE", "1"))
        ga = args.gradient_accumulation if args.gradient_accumulation is not None else 1
        args.total_batch_size = args.batch_size * ws * ga

    if args.gradient_accumulation is None:
        ws = 1 if args.single_gpu else int(os.environ.get("WORLD_SIZE", "1"))
        args.gradient_accumulation = max(1, args.total_batch_size // (args.batch_size * ws))

    # final consistency
    ws = 1 if args.single_gpu else int(os.environ.get("WORLD_SIZE", "1"))
    assert args.gradient_accumulation * args.batch_size * ws == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must equal total_batch_size"

    # build final run name
    args.run_name = build_run_name(args)
    return args


# -------------------------
# Distributed setup
# -------------------------

def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
    """
    Returns (global_rank, local_rank, world_size, device).
    """
    if args.single_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return 0, 0, 1, device

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")
    return global_rank, local_rank, world_size, device


# -------------------------
# Data / Model
# -------------------------

def load_train_data(args: argparse.Namespace) -> Any:
    if args.dataset_name == "openwebtext":
        ds = datasets.load_dataset(args.dataset_path, split="train", trust_remote_code=True)
        ds = ds.train_test_split(test_size=0.05, seed=args.seed)
        return ds, ds["train"]
    elif args.dataset_name == "c4":
        while True:
            try:
                train = datasets.load_dataset("allenai/c4", "en", split="train", trust_remote_code=True)
                return None, train
            except Exception as e:
                if rank0_only():
                    print(f"Error loading dataset: {e}")
                time.sleep(3)
    raise ValueError(f"Unknown dataset_name: {args.dataset_name}")


def load_val_data(args: argparse.Namespace, train_valid_data: Any) -> Any:
    if args.dataset_name == "openwebtext":
        assert train_valid_data is not None
        return train_valid_data["test"], ["text"]
    elif args.dataset_name == "c4":
        while True:
            try:
                val = datasets.load_dataset("allenai/c4", "en", split="validation", trust_remote_code=True)
                return val, ["text", "timestamp", "url"]
            except Exception as e:
                if rank0_only():
                    print(f"Error loading validation dataset: {e}")
                time.sleep(5)
    raise ValueError(f"Unknown dataset_name: {args.dataset_name}")


def build_tokenizer(args: argparse.Namespace) -> Any:
    while True:
        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.max_length, trust_remote_code=True)
            if tok.pad_token_id is None:
                # Many tokenizers don't have pad_token by default
                tok.pad_token = tok.eos_token
            return tok
        except Exception as e:
            if rank0_only():
                print(f"Error loading tokenizer: {e}")
            time.sleep(5)


def build_model(args: argparse.Namespace, model_config: Any) -> torch.nn.Module:
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    return model


def move_model_to_device(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if args.dtype in ["bf16", "bfloat16"]:
        return model.to(device=device, dtype=torch.bfloat16)
    return model.to(device=device)


def maybe_wrap_ddp(args: argparse.Namespace, model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    if args.single_gpu:
        return model

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    return model


# -------------------------
# Eval
# -------------------------

@torch.no_grad()
def evaluate_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    preprocess_batched,
    device: torch.device,
    global_rank: int,
    world_size: int,
    batch_size: int,
    train_valid_data: Any,
) -> Tuple[float, int]:
    t0 = time.time()

    val_data, remove_columns = load_val_data(args, train_valid_data)
    val_data = val_data.shuffle(seed=42)

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    # IMPORTANT: return python lists (HF datasets map-safe), not torch tensors
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=remove_columns,
    )
    val_data_mapped.batch = lambda bs: training_utils.batch_fn(val_data_mapped, bs)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0, device=device)
    total_batches = 0

    pad_idx = tokenizer.pad_token_id

    if rank0_only():
        logger.info(f"Eval set prepared in {time.time() - t0:.2f}s")
        
    def _batch(*, batch_size: int, **_):
        return training_utils.batch_fn(val_data_mapped, batch_size)

    val_data_mapped.batch = _batch
    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break

        # Convert to tensors here
        batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}

        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()
        total_batches += 1

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    if total_batches == 0:
        return float("nan"), 0

    total_loss = total_loss / total_batches

    # Gather losses across ranks if DDP
    if is_dist_initialized():
        gathered = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered, total_loss)
        avg_loss = sum([t.item() for t in gathered]) / world_size
    else:
        avg_loss = total_loss.item()

    return avg_loss, evaluated_on_tokens


# -------------------------
# Checkpointing
# -------------------------

def checkpoint_dir(args: argparse.Namespace, model_name: str, update_step: int) -> str:
    if args.only_save_last:
        return args.save_dir
    return os.path.join(args.save_dir, f"model_{update_step}")


def save_checkpoint(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    pruner: Optional[Any],
    run_config: Dict[str, Any],
    global_step: int,
    update_step: int,
    tokens_seen: int,
    tokens_seen_before: int,
    update_time: float,
) -> None:
    if not rank0_only():
        return

    out_dir = checkpoint_dir(args, "model", update_step)
    os.makedirs(out_dir, exist_ok=True)

    # model state
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(out_dir, "all_model.pt"))

    # optimizer/scheduler
    opt_payload = {
        "optimizer": optimizer.state_dict(),
        "scheduler": getattr(scheduler, "state_dict", lambda: {})(),
        "dst_scheduler": pruner.state_dict() if (pruner is not None and hasattr(pruner, "state_dict")) else None,
        "update_step": update_step,
        "global_step": global_step,
        "config": run_config,
        "wandb_dir": wandb.run.dir if wandb.run is not None else None,
        "dtype": args.dtype,
    }
    torch.save(opt_payload, os.path.join(out_dir, "optimizer.pt"))

    # training state json
    train_state = {
        "global_step": global_step,
        "update_step": update_step,
        "tokens_seen": tokens_seen,
        "tokens_seen_before": tokens_seen_before,
        "update_time": update_time,
    }
    with open(os.path.join(out_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(train_state, f, indent=2)

    # wandb id for resume
    wandb_info = {"wandb_id": wandb.run.id if wandb.run is not None else None}
    with open(os.path.join(args.save_dir, "wandb.json"), "w", encoding="utf-8") as f:
        json.dump(wandb_info, f, indent=2)


def load_checkpoint_if_needed(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    pruner: Optional[Any],
) -> Tuple[int, int, int, int]:
    """
    Returns (global_step, update_step, tokens_seen, tokens_seen_before).
    """
    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is None:
        return global_step, update_step, tokens_seen, tokens_seen_before

    ckpt_dir = args.continue_from
    logger.info(f"Loading checkpoint from {ckpt_dir}")

    model_path = os.path.join(ckpt_dir, "all_model.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    # Strip possible DDP prefix
    prefix = "module."
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logger.info("Model state restored (strict=True)")

    train_state_path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(train_state_path):
        with open(train_state_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        global_step = int(old.get("global_step", 0))
        update_step = int(old.get("update_step", 0))
        tokens_seen = int(old.get("tokens_seen", 0))
        tokens_seen_before = int(old.get("tokens_seen_before", 0))
        logger.info(f"Restored training state: global_step={global_step}, update_step={update_step}")
    else:
        logger.warning("training_state.json not found; steps reset to 0")

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    opt_payload = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(opt_payload["optimizer"])

    if "scheduler" in opt_payload and opt_payload["scheduler"] is not None and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(opt_payload["scheduler"])
        except Exception as e:
            logger.warning(f"Failed to restore scheduler state: {e}")

    if args.dst_scheduler and pruner is not None and opt_payload.get("dst_scheduler") is not None:
        try:
            pruner.load_state_dict(opt_payload["dst_scheduler"])
        except Exception as e:
            logger.warning(f"Failed to restore DST scheduler state: {e}")

    logger.info("Optimizer (and scheduler/pruner if available) restored")
    return global_step, update_step, tokens_seen, tokens_seen_before


# -------------------------
# Main training
# -------------------------

def main(args: argparse.Namespace) -> None:
    # perf toggles (safe defaults)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    global_rank, local_rank, world_size, device = setup_distributed(args)

    # rank != 0: silence loguru output
    if not rank0_only():
        logger.remove()

    set_seed(args.seed)

    # optional file logging (rank0)
    run_tag = f"train_{args.dataset_name}_{os.path.splitext(os.path.basename(args.model_config))[0]}_{args.run_name}"
    log_file = setup_logging_to_file(args.log_to_file, args.log_dir, run_tag)

    logger.info(f"Rank {global_rank}/{world_size} on device={device}")

    model_name = os.path.splitext(os.path.basename(args.model_config))[0]
    args.save_dir = args.save_dir or os.path.join("runs", f"{args.dataset_name}-{model_name}", args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # Prevent accidental overwrite of final export dir
    final_export_dir = os.path.join("trained_model", f"galore-{args.dataset_name}-{model_name}_{args.run_name}")
    if os.path.exists(final_export_dir):
        if rank0_only():
            logger.warning(f"Final export dir exists: {final_export_dir}. Will not overwrite; exiting.")
        safe_barrier()
        if log_file is not None:
            sys.stdout = sys.__stdout__
            log_file.close()
        return

    # wandb
    if rank0_only():
        wandb.init(
            project=f"galore-{args.dataset_name}",
            name=f"{model_name}_{args.run_name}",
            mode="disabled" if args.no_log else "online",
        )

    # data
    train_valid_data, data = load_train_data(args)
    data = data.shuffle(seed=42)

    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(data, rank=global_rank, world_size=world_size)

    tokenizer = build_tokenizer(args)
    pad_idx = tokenizer.pad_token_id

    def preprocess_batched(batch: Dict[str, Any]) -> Dict[str, Any]:
        # HF datasets "map" safe output (lists), no torch tensors here.
        enc = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        return enc

    # train iterable dataset (your class handles tokenization internally)
    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=args.workers)

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = build_model(args, model_config)
    model = move_model_to_device(args, model, device)

    # optional grad checkpoint
    if getattr(args, "activation_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # scheduler (keep your training_utils call, but add fallback)
    try:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            restart_warmup_steps=args.iterative_warmup_steps,
            cycle_length=args.update_interval,
            no_decay=args.no_decay,
        )
    except AttributeError:
        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            restart_warmup_steps=args.iterative_warmup_steps,
            cycle_length=args.update_interval,
            no_decay=args.no_decay,
        )

    # pruning schedule
    if args.pruning_T_end is None:
        args.pruning_T_end = int(args.num_training_steps * 0.75)

    pruner = None
    if args.dst_scheduler:
        pruner = DSTScheduler(
            model,
            optimizer,
            alpha=args.zeta,
            delta=args.update_interval,
            static_topo=False,
            T_end=int(args.num_training_steps * 0.75),
            ignore_linear_layers=False,
            sparsity_distribution=args.sparsity_distribution,
            grad_accumulation_n=args.gradient_accumulation,
            args=args,
        )

    # load checkpoint if needed (before DDP wrap)
    global_step, update_step, tokens_seen, tokens_seen_before = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, pruner
    )

    # DDP
    model = maybe_wrap_ddp(args, model, local_rank)

    # generation config pad token safety
    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = pad_idx

    # wandb config / script save
    n_total_params = sum(p.numel() for p in base_model.parameters())
    run_config = dict(vars(args))
    run_config.update({
        "total_params_M": n_total_params / 1_000_000,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "screen_id": os.environ.get("STY", "no-screen"),
    })

    if rank0_only():
        wandb.config.update(run_config, allow_val_change=True)
        try:
            wandb.save(os.path.abspath(__file__), policy="now")
        except Exception:
            pass

    # log args
    logger.info("*" * 60)
    logger.info("Arguments:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"{k:28} {v}")
    logger.info("*" * 60)

    if rank0_only():
        pbar = tqdm(total=max(0, args.num_training_steps - update_step), desc="Update steps", ncols=90)

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    model.train()
    update_time_start = time.time()
    local_step = 0

    for _, batch in enumerate(dataloader):
        global_step += 1
        local_step += 1

        if update_step >= args.num_training_steps:
            logger.info(f"Reached num_training_steps={args.num_training_steps}; stopping.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        (loss / args.gradient_accumulation).backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # UPDATE STEP
        if args.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if args.dst_scheduler and pruner is not None:
            if pruner():
                optimizer.step()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        update_step += 1
        update_time = time.time() - update_time_start

        if rank0_only():
            pbar.update(1)

        # save
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            save_checkpoint(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                pruner=pruner,
                run_config=run_config,
                global_step=global_step,
                update_step=update_step,
                tokens_seen=tokens_seen,
                tokens_seen_before=tokens_seen_before,
                update_time=update_time,
            )

        # eval
        if update_step % args.eval_every == 0:
            logger.info(f"Eval at update_step={update_step}")
            model.eval()
            avg_loss, eval_tokens = evaluate_model(
                args=args,
                model=model,
                tokenizer=tokenizer,
                preprocess_batched=preprocess_batched,
                device=device,
                global_rank=global_rank,
                world_size=world_size,
                batch_size=args.batch_size,
                train_valid_data=train_valid_data,
            )
            model.train()

            if rank0_only():
                wandb.log(
                    {"eval_loss": avg_loss, "eval_tokens": eval_tokens},
                    step=update_step,
                )
                if args.no_log:
                    print(f"[eval] step={update_step} loss={avg_loss} tokens={eval_tokens}")

            logger.info(f"Eval loss={avg_loss}")

        # train logging
        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen

        if rank0_only():
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / max(update_time, 1e-9),
                    "throughput_examples": args.total_batch_size / max(update_time, 1e-9),
                },
                step=update_step,
            )
            if args.no_log:
                print(f"[train] step={update_step} loss={loss.item():.4f} lr={lr:.3e}")

        update_time_start = time.time()

    # -------------------------
    # END TRAIN LOOP
    # -------------------------
    logger.info("Training finished")
    if rank0_only():
        pbar.close()

    safe_barrier()

    # export pretrained format (rank0)
    if rank0_only():
        os.makedirs(os.path.dirname(final_export_dir), exist_ok=True)
        base_model = model.module if hasattr(model, "module") else model
        base_model.save_pretrained(final_export_dir)
        tokenizer.save_pretrained(final_export_dir)
        logger.info(f"Saved final model to: {final_export_dir}")

    # final eval (optional)
    logger.info("Running final evaluation")
    model.eval()
    torch.cuda.empty_cache()

    avg_loss, eval_tokens = evaluate_model(
        args=args,
        model=model,
        tokenizer=tokenizer,
        preprocess_batched=preprocess_batched,
        device=device,
        global_rank=global_rank,
        world_size=world_size,
        batch_size=args.batch_size,
        train_valid_data=train_valid_data,
    )

    if rank0_only():
        wandb.log({"final_eval_loss": avg_loss, "final_eval_tokens": eval_tokens}, step=update_step)
        logger.info(f"Final eval loss: {avg_loss}")

    logger.info("Script finished successfully")

    # cleanup
    safe_barrier()
    if is_dist_initialized():
        dist.destroy_process_group()

    if log_file is not None and rank0_only():
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    print("Starting script")
    args = parse_args()
    main(args)
