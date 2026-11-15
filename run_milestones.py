#!/usr/bin/env python3
"""Run benchmarks at historical Crowd-v1 global steps divisible by 20."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, env=env, text=True).strip()


def milestones(repo: Path, start_step: int, end_step: int | None) -> list[tuple[int, str]]:
    log = run(
        ["git", "log", "--all", "--format=%H%x09%s"],
        cwd=repo,
    )
    found: dict[int, str] = {}
    for line in log.splitlines():
        commit, _, title = line.partition("\t")
        match = re.search(r"\bUpdate step (\d+)\b", title)
        if not match:
            continue
        step = int(match.group(1))
        if step >= start_step and step % 20 == 0 and (end_step is None or step <= end_step):
            found.setdefault(step, commit)
    return sorted(found.items(), reverse=True)


def cleanup_weights(repo: Path, revision: str) -> None:
    for filename in ("model_bf16.safetensors", "engram_bf16.safetensors"):
        pointer = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{filename}"], text=True
        )
        match = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
        if match:
            oid = match.group(1)
            obj = repo / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            obj.unlink(missing_ok=True)
        (repo / filename).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/crowd-eval/results"))
    parser.add_argument("--start-step", type=int, default=280)
    parser.add_argument("--end-step", type=int, default=360)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--tasks",
        default="hellaswag,piqa,arc_easy,winogrande,lambada_openai,mmlu",
    )
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        bundled_tokenizer = Path(__file__).resolve().parent / "tokenizer"
        if bundled_tokenizer.is_dir():
            tokenizer_dir = bundled_tokenizer
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    selected = milestones(repo, args.start_step, args.end_step)
    if not selected:
        raise SystemExit("No checkpoint commits found at requested 20-step intervals")
    print(f"Found {len(selected)} checkpoints: {selected[0][0]}..{selected[-1][0]}", flush=True)

    for step, commit in selected:
        existing = list(output.glob(f"step_{step:04d}_{commit[:8]}.json"))
        if existing:
            print(f"Skipping step {step}: result already exists", flush=True)
            continue
        print(f"\n=== step {step}: {commit} ===", flush=True)
        try:
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--detach", commit],
                env=env,
                check=True,
            )
            subprocess.run(
                [
                    "git", "-C", str(repo), "lfs", "pull",
                    "--include=model_bf16.safetensors,engram_bf16.safetensors,tokenizer.json,vocab.json",
                    "--exclude=",
                ],
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Skipping step {step}: checkout/download failed ({exc.returncode})", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).with_name("eval_checkpoint.py")),
            "--model-dir", str(repo),
            "--code-dir", str(Path(__file__).resolve().parent / "model_code"),
            "--output-dir", str(output),
            "--tasks", args.tasks,
            "--batch-size", str(args.batch_size),
        ]
        if tokenizer_dir is not None:
            command.extend(["--tokenizer-dir", str(tokenizer_dir.resolve())])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Step {step} evaluation failed ({exc.returncode}); continuing", flush=True)
        finally:
            cleanup_weights(repo, commit)

    print(f"Finished. Results are in {output}", flush=True)


if __name__ == "__main__":
    main()
