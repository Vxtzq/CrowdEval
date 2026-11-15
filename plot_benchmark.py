#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

# ==========================================
# CONFIGURATION
# ==========================================
TASKS = ["mmlu", "hellaswag", "piqa", "arc_easy", "winogrande", "lambada_openai"]
OUTPUT_PATH = "results/qwen_style_benchmark_grid.png"

# Base-model scores from Qwen official reports
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

# You can download small PNG/SVG logos and put their paths here
# If you don't have logos, the script will just use text labels.
MODEL_LOGOS = {
    "Crowd-v1": None, # e.g., "assets/crowd_logo.png"
    "Qwen2.5-0.5B": None,
    "Qwen3-0.6B": None,
    "Qwen3.5-0.8B": None,
}

def get_primary_metric(metrics: dict) -> float:
    for k in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if k in metrics: return metrics[k]
    return 0.0

def main():
    # 1. Load the latest Crowd-v1 result
    json_files = glob.glob("results/json/*.json")
    if not json_files:
        print("Error: No JSON files found in results/json/.")
        return
    
    latest_file = max(json_files, key=lambda f: Path(f).stat().st_mtime)
    with open(latest_file, "r") as f:
        res = json.load(f)
        
    step = res["step"]
    crowd_name = f"Crowd-v1 (Step {step})"
    
    # 2. Build the DataFrame
    data = []
    for task in TASKS:
        # Crowd
        if task in res["results"]:
            data.append({"Model": crowd_name, "Task": task, "Score": get_primary_metric(res["results"][task])})
        # Qwen
        for qwen_name, qwen_scores in QWEN_BASELINES.items():
            data.append({"Model": qwen_name, "Task": task, "Score": qwen_scores[task]})
            
    df = pd.DataFrame(data)

    # 3. Create the Grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    sns.set_theme(style="whitegrid")
    
    # Define colors: Blue for Crowd, Grays/Oranges for Qwen
    palette = {
        crowd_name: "#2E86AB", 
        "Qwen2.5-0.5B": "#E63946", 
        "Qwen3-0.6B": "#F4A261", 
        "Qwen3.5-0.8B": "#E9C46A"
    }

    for i, task in enumerate(TASKS):
        ax = axes[i]
        task_df = df[df["Task"] == task]
        
        # Draw the bars
        sns.barplot(
            data=task_df, x="Model", y="Score", 
            ax=ax, palette=palette, width=0.6
        )
        
        # Formatting
        ax.set_title(task.upper().replace("_", "-"), fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel("Accuracy" if i % 3 == 0 else "")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.0)
        
        # Rotate x-labels so they fit
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        # Remove top and right spines for that clean look
        sns.despine(ax=ax)

        # Add value labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', 
                            xytext=(0, 3), textcoords='offset points',
                            fontsize=10, fontweight="bold")

        # Optional: Overlay Logos if provided
        for j, model in enumerate(task_df["Model"].unique()):
            if MODEL_LOGOS.get(model):
                try:
                    # Load image and put it above the bar
                    logo = mpimg.imread(MODEL_LOGOS[model])
                    ax.imshow(logo, extent=[j-0.15, j+0.15, 0.95, 1.05], aspect='auto', zorder=10)
                except Exception as e:
                    pass # Fail silently if logo missing

    # 4. Global Formatting
    fig.suptitle(f"Crowd-v1 vs Qwen Baselines (Model Size ~0.5B - 0.8B)", 
                 fontsize=20, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # 5. Save
    Path("results").mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Benchmarks grid saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
