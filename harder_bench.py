#!/usr/bin/env python3
"""Evaluate Crowd-v1 on hard generative benchmarks: GSM8K and GPQA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

def get_flat(filename: Path) -> torch.Tensor:
    tensors = load_file(str(filename), device="cpu")
    if "weights" in tensors:
        tensor = tensors["weights"]
    elif "flat" in tensors:
        tensor = tensors["flat"]
    else:
        tensor = next(iter(tensors.values()))
    return tensor.contiguous().view(-1)

def load_crowd_model(repo: Path, code_dir: Path) -> tuple[torch.nn.Module, object]:
    sys.path.insert(0, str(code_dir))
    from configuration_crowd import CrowdConfig
    from modeling_crowd import CrowdForCausalLM

    raw_config = json.loads((repo / "config.json").read_text())
    if "model_type" in raw_config:
        config = CrowdConfig(
            vocab_size=raw_config["vocab_size"], hidden_size=raw_config["hidden_size"],
            num_hidden_layers=raw_config["num_hidden_layers"], num_attention_heads=raw_config["num_attention_heads"],
            num_key_value_heads=raw_config["num_key_value_heads"], head_dim=raw_config["head_dim"],
            intermediate_size=raw_config["intermediate_size"], max_position_embeddings=raw_config["max_position_embeddings"],
            hidden_act=raw_config["hidden_act"], layer_norm_eps=raw_config["layer_norm_eps"],
            rope_theta=raw_config["rope_theta"], attention_bias=raw_config["attention_bias"],
            mlp_bias=raw_config["mlp_bias"], tie_word_embeddings=raw_config["tie_word_embeddings"],
            engram_buckets=raw_config["engram_buckets"], initializer_range=raw_config["initializer_range"],
        )
    else:
        config = CrowdConfig(
            vocab_size=raw_config["vocabSize"], hidden_size=raw_config["dim"],
            num_hidden_layers=raw_config["nLayers"], num_attention_heads=raw_config["nHeads"],
            num_key_value_heads=raw_config["nKvHeads"], head_dim=raw_config["headDim"],
            intermediate_size=raw_config["mlpHidden"], max_position_embeddings=raw_config["maxSeqLen"],
            tie_word_embeddings=raw_config["weightTying"],
        )
    model = CrowdForCausalLM(config)
    model.to(dtype=torch.bfloat16)

    flat = get_flat(repo / "model_bf16.safetensors")
    offset = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith("engram."): continue
            end = offset + param.numel()
            if end > flat.numel(): raise ValueError(f"Flat model ended at {name}")
            param.copy_(flat[offset:end].view_as(param))
            offset = end
    del flat

    engram = get_flat(repo / "engram_bf16.safetensors")
    table = model.engram.table.weight
    with torch.no_grad():
        table.copy_(engram.view_as(table))
    del engram

    tokenizer = AutoTokenizer.from_pretrained(
        str(repo), trust_remote_code=True, local_files_only=True, use_fast=True, fix_mistral_regex=True
    )
    model.eval().to("cuda")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Limit examples to speed up testing (default 100)")
    parser.add_argument("--output-dir", type=Path, default=Path("./results/hard_json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Auto-downloading Vxtzq/Crowd-v1 from Hugging Face...")
    repo_path = snapshot_download(
        repo_id="Vxtzq/Crowd-v1",
        allow_patterns=["config.json", "training_state.json", "model_bf16.safetensors", "engram_bf16.safetensors", "*.py", "*.model", "tokenizer*"]
    )
    repo = Path(repo_path)

    config_data = json.loads((repo / "config.json").read_text())
    state_file = repo / "training_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else config_data
    step = int(state.get("globalStep", config_data.get("globalStep", 0)))

    print(f"Loading Crowd-v1 step {step}...")
    model, tokenizer = load_crowd_model(repo, repo)
    print(f"GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # We use HFLM with max_gen_toks for generative tasks
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=int(model.config.max_position_embeddings),
        max_gen_toks=256, # Model needs to generate reasoning for GSM8K
        logits_cache=True,
    )
    
    # GPQA is 0-shot, GSM8K is 5-shot by default in lm-eval
    tasks = ["gsm8k", "gpqa_main_zeroshot"]
    
    print(f"Evaluating {tasks} with limit={args.limit}...")
    result = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=args.limit,
        batch_size=4,
        cache_requests=True,
        log_samples=True, # Save samples so you can see the 0.1% answers!
        random_seed=0,
    )

    payload = {
        "model": "Vxtzq/Crowd-v1",
        "step": step,
        "results": result.get("results", {}),
        "samples": result.get("samples", {}), # This will show exactly what it generated
    }
    
    out = args.output_dir / f"step_{step:04d}_hard.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    
    print("\n" + "="*50)
    print(f"HARD BENCHMARK RESULTS (Limit: {args.limit})")
    print("="*50)
    for task, metrics in payload["results"].items():
        print(f"[{task.upper()}]")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    
    print(f"\n✅ Saved raw generations to {out}")
    print("Look inside that JSON under 'samples' to see the funny outputs.")

if __name__ == "__main__":
    main()
