#!/usr/bin/env python3
"""Evaluate one Crowd-v1 flat-weight checkpoint with lm-evaluation-harness."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def lfs_oid(repo: Path, filename: str, revision: str) -> str:
    pointer = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{revision}:{filename}"], text=True
    )
    match = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    if not match:
        raise ValueError(f"No LFS SHA-256 in {revision}:{filename}")
    return match.group(1)


def get_flat(filename: Path) -> torch.Tensor:
    tensors = load_file(str(filename), device="cpu")
    if "weights" in tensors:
        tensor = tensors["weights"]
    elif "flat" in tensors:
        tensor = tensors["flat"]
    else:
        tensor = next(iter(tensors.values()))
    return tensor.contiguous().view(-1)


def load_crowd_model(
    repo: Path, code_dir: Path, tokenizer_dir: Path | None = None
) -> tuple[torch.nn.Module, object, dict[str, str]]:
    sys.path.insert(0, str(code_dir))
    from configuration_crowd import CrowdConfig
    from modeling_crowd import CrowdForCausalLM

    raw_config = json.loads((repo / "config.json").read_text())
    if "model_type" in raw_config:
        config = CrowdConfig(
            vocab_size=raw_config["vocab_size"],
            hidden_size=raw_config["hidden_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=raw_config["num_attention_heads"],
            num_key_value_heads=raw_config["num_key_value_heads"],
            head_dim=raw_config["head_dim"],
            intermediate_size=raw_config["intermediate_size"],
            max_position_embeddings=raw_config["max_position_embeddings"],
            hidden_act=raw_config["hidden_act"],
            layer_norm_eps=raw_config["layer_norm_eps"],
            rope_theta=raw_config["rope_theta"],
            attention_bias=raw_config["attention_bias"],
            mlp_bias=raw_config["mlp_bias"],
            tie_word_embeddings=raw_config["tie_word_embeddings"],
            engram_buckets=raw_config["engram_buckets"],
            initializer_range=raw_config["initializer_range"],
        )
    else:
        # Early training commits use CrowdGPT's original camelCase config.
        config = CrowdConfig(
            vocab_size=raw_config["vocabSize"],
            hidden_size=raw_config["dim"],
            num_hidden_layers=raw_config["nLayers"],
            num_attention_heads=raw_config["nHeads"],
            num_key_value_heads=raw_config["nKvHeads"],
            head_dim=raw_config["headDim"],
            intermediate_size=raw_config["mlpHidden"],
            max_position_embeddings=raw_config["maxSeqLen"],
            tie_word_embeddings=raw_config["weightTying"],
        )
    model = CrowdForCausalLM(config)
    model.to(dtype=torch.bfloat16)

    flat = get_flat(repo / "model_bf16.safetensors")
    offset = 0
    loaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith("engram."):
                continue
            end = offset + param.numel()
            if end > flat.numel():
                raise ValueError(f"Flat model weights ended while loading {name}")
            param.copy_(flat[offset:end].view_as(param))
            offset = end
            loaded += param.numel()
    if offset != flat.numel():
        raise ValueError(f"Loaded {offset} of {flat.numel()} base-model values")
    del flat

    engram = get_flat(repo / "engram_bf16.safetensors")
    table = model.engram.table.weight
    if engram.numel() != table.numel():
        raise ValueError(
            f"Engram size mismatch: checkpoint={engram.numel()} model={table.numel()}"
        )
    with torch.no_grad():
        table.copy_(engram.view_as(table))
    del engram

    tokenizer_path = tokenizer_dir or repo
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
        fix_mistral_regex=True,
    )
    model.eval().to("cuda")

    rev = git_text(repo, "rev-parse", "HEAD")
    metadata = {
        "revision": rev,
        "model_bf16_sha256": lfs_oid(repo, "model_bf16.safetensors", rev),
        "engram_bf16_sha256": lfs_oid(repo, "engram_bf16.safetensors", rev),
        "parameter_count_loaded": str(loaded),
        "model_code_revision": "ebf5e1acff6c799079d7517272afa30ee854b80c",
    }
    return model, tokenizer, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument(
        "--code-dir", type=Path, default=Path(__file__).resolve().parent / "model_code"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="hellaswag,piqa,arc_easy,winogrande,lambada_openai,mmlu",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run a single evaluation and save a concise markdown summary to results/benchmark.txt"
    )
    args = parser.parse_args()

    repo = args.model_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_data = json.loads((repo / "config.json").read_text())
    state_file = repo / "training_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else config_data
    step = int(state.get("globalStep", config_data.get("globalStep")))

    print(f"Loading Crowd-v1 step {step} from {repo}", flush=True)
    tokenizer_dir = args.tokenizer_dir.resolve() if args.tokenizer_dir else None
    model, tokenizer, metadata = load_crowd_model(
        repo, args.code_dir.resolve(), tokenizer_dir
    )
    print(
        f"Loaded revision {metadata['revision']} ({metadata['parameter_count_loaded']} base params); "
        f"GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )

    # lm-eval right-pads causal batches. The causal attention mask prevents those
    # trailing pad tokens from affecting real-token logits, so vectorized scoring
    # remains equivalent to batch size one.
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=int(model.config.max_position_embeddings),
        logits_cache=True,
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    print(f"Evaluating {tasks}; limit={args.limit}", flush=True)
    result = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        cache_requests=True,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )

    payload = {
        "model": "Vxtzq/Crowd-v1",
        "model_kind": "base",
        "step": step,
        "checkpoint": metadata,
        "tokenizer_dir": str(tokenizer_dir or repo),
        "tasks": tasks,
        "num_fewshot_override": args.num_fewshot,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "harness_version": "0.4.13",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "results": result.get("results", {}),
        "task_configs": result.get("config", {}),
        "versions": result.get("versions", {}),
    }
    out = args.output_dir / f"step_{step:04d}_{metadata['revision'][:8]}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"Saved {out}", flush=True)

    # Generate console output AND prepare the markdown for the benchmark file
    md_lines = []
    md_lines.append(f"# Crowd-v1 Benchmark (Step {step})")
    md_lines.append(f"**Revision**: `{metadata['revision']}`")
    md_lines.append(f"**Params**: {metadata['parameter_count_loaded']}")
    md_lines.append("")
    md_lines.append("| Task | Metric | Value |")
    md_lines.append("|---|---|---|")

    for task, metrics in payload["results"].items():
        preferred = [k for k in metrics if k.startswith("acc_norm,") or k == "acc_norm"]
        if not preferred:
            preferred = [k for k in metrics if k.startswith("acc,") or k == "acc"]
        if not preferred:
            preferred = list(metrics)
        
        # Console print (as before, showing top 2)
        values = ", ".join(f"{key}={metrics[key]}" for key in preferred[:2])
        print(f"{task}: {values}", flush=True)

        # Markdown format (taking the primary metric)
        main_key = preferred[0]
        md_lines.append(f"| {task} | {main_key} | {metrics[main_key]:.4f} |")

    # Save benchmark.txt if requested
    if args.benchmark:
        bench_dir = Path("results")
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_file = bench_dir / "benchmark.txt"
        
        # If the file exists, append; otherwise write header
        mode = "a" if bench_file.exists() else "w"
        with open(bench_file, mode) as f:
            if mode == "a":
                f.write("\n\n---\n\n")
            f.write("\n".join(md_lines) + "\n")
            
        print(f"\n✅ Benchmark appended to {bench_file.absolute()}", flush=True)


if __name__ == "__main__":
    main()
