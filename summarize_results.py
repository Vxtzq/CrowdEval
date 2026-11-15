#!/usr/bin/env python3
"""Create a compact CSV from per-checkpoint lm-eval JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = {
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "winogrande": "acc,none",
    "lambada_openai": "acc,none",
    "mmlu": "acc,none",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.results_dir / "summary.csv")

    rows = []
    for path in sorted(args.results_dir.glob("step_*.json")):
        payload = json.loads(path.read_text())
        results = payload.get("results", {})
        row = {
            "step": payload.get("step"),
            "revision": payload.get("checkpoint", {}).get("revision"),
            "result_file": path.name,
        }
        for task, metric in TASKS.items():
            task_result = results.get(task, {})
            value = task_result.get(metric)
            if value is None:
                # Harness versions may spell the aggregate metric without its aggregation suffix.
                value = task_result.get(metric.split(",")[0])
            row[task] = value
        rows.append(row)
    rows.sort(key=lambda item: item["step"] or -1)
    columns = ["step", "revision", *TASKS.keys(), "result_file"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checkpoints to {output}")

    if rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        for task in TASKS:
            points = [
                (row["step"], float(row[task]) * 100)
                for row in rows
                if row[task] is not None
            ]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", linewidth=1.8, markersize=5, label=task)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Crowd-v1 base-model benchmark scores")
        ax.set_xticks([row["step"] for row in rows])
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        plot_path = args.results_dir / "score_by_step.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
