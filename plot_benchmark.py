#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# BASELINE CONFIG
# ==========================================
TASKS = ["mmlu", "hellaswag", "piqa", "arc_easy", "winogrande", "lambada_openai"]
OUTPUT_PATH = "results/benchmark_latest_vs_qwen.png"

# Base-model scores from Qwen official reports/evaluations
QWEN_BASELINES = {
    "Qwen2.5-0.5B": {
        "mmlu": 0.272, "hellaswag": 0.477, "piqa": 0.660, 
        "arc_easy": 0.646, "winogrande": 0.540, "lambada_openai": 0.395
    },
    "Qwen3-0.6B": {
        "mmlu": 0.435, "hellaswag": 0.403, "piqa": 0.570, 
        "arc_easy": 0.600, "winogrande": 0.507, "lambada_openai": 0.418
    },
    "Qwen3.5-0.8B": {
        "mmlu": 0.308, "hellaswag": 0.389, "piqa": 0.682, 
        "arc_easy": 0.462, "winogrande": 0.573, "lambada_openai": 0.360
    }
}

def get_primary_metric(metrics: dict) -> float:
    """Extract the primary accuracy metric from lm-eval JSON."""
    for k in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if k in metrics:
            return metrics[k]
    return 0.0

def main():
    # 1. Find the latest JSON result
    json_files = glob.glob("results/json/*.json")
    if not json_files:
        print("Error: No JSON files found in results/json/. Run evaluate.py first.")
        return
    
    # Sort by modification time to get the latest one
    latest_file = max(json_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading latest checkpoint: {latest_file}")
    
    with open(latest_file, "r") as f:
        res = json.load(f)
        
    step = res["step"]
    model_name = f"Crowd-v1 (Step {step})"
    
    # 2. Build the DataFrame
    data = []
    
    # Add Crowd-v1 results
    for task in TASKS:
        if task in res["results"]:
            score = get_primary_metric(res["results"][task])
            data.append({"Model": model_name, "Task": task.upper(), "Score": score})
        else:
            print(f"Warning: {task} not found in checkpoint results.")
            
    # Add Qwen baselines
    for qwen_name, qwen_scores in QWEN_BASELINES.items():
        for task in TASKS:
            data.append({
                "Model": qwen_name, 
                "Task": task.upper(), 
                "Score": qwen_scores[task]
            })
            
    df = pd.DataFrame(data)
    
    # 3. Plotting
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Order the x-axis so it looks logical
    task_order = [t.upper() for t in TASKS]
    
    # Use a custom palette: Blue for Crowd, Red/Orange for Qwen
    palette = {
        model_name: "#2E86AB", 
        "Qwen2.5-0.5B": "#E63946", 
        "Qwen3-0.6B": "#F4A261", 
        "Qwen3.5-0.8B": "#E9C46A"
    }
    
    ax = sns.barplot(
        data=df, x="Task", y="Score", hue="Model", 
        order=task_order, palette=palette
    )
    
    # 4. Formatting
    plt.title(f"Crowd-v1 (Step {step}) vs Qwen Baselines", fontsize=18, fontweight="bold", pad=15)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Benchmark Task", fontsize=14)
    plt.ylim(0, 1.0)
    
    # Move legend outside
    plt.legend(title="Model", title_fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=9, fontweight="bold", rotation=45)
            
    plt.tight_layout()
    
    # 5. Save
    Path("results").mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"✅ Chart successfully saved to {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
