#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
TASKS = ["mmlu", "hellaswag", "piqa", "arc_easy", "winogrande", "lambada_openai"]
TASK_TITLES = {
    "mmlu": ("MMLU", "General Knowledge"),
    "hellaswag": ("HellaSwag", "Commonsense Reasoning"),
    "piqa": ("PIQA", "Physical Interaction"),
    "arc_easy": ("ARC-Easy", "Science Questions"),
    "winogrande": ("WinoGrande", "Coreference Resolution"),
    "lambada_openai": ("LAMBADA", "Word Prediction")
}

OUTPUT_PATH = "results/qwen_style_benchmark_grid.png"

# Base-model scores
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

# CONFIGURE LOGOS HERE
LOGOS = {
    "Crowd-v1": "assets/logos/crowd.png",
    "Qwen2.5-0.5B": "assets/logos/qwen25.png",
    "Qwen3-0.6B": "assets/logos/qwen3.png",
    "Qwen3.5-0.8B": "assets/logos/qwen35.png",
}

# Colors based on your screenshot
COLORS = {
    "Crowd-v1": "#5B6CFF",      # Bright Blue
    "Qwen2.5-0.5B": "#A0A6D4",  # Light Blue-Gray
    "Qwen3-0.6B": "#C0C4D6",    # Lighter Gray
    "Qwen3.5-0.8B": "#D0D3E0"   # Lightest Gray
}

def get_primary_metric(metrics: dict) -> float:
    for k in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if k in metrics: return metrics[k]
    return 0.0

def draw_bar(ax, x_pos, height, color, value, logo_path=None):
    """Draws a single bar with value text and a logo."""
    # Draw the bar
    ax.bar(x_pos, height, width=0.6, color=color, zorder=2)
    
    # Add value text on top
    ax.text(x_pos, height + 1.5, f"{value:.1f}", ha='center', va='bottom', 
            fontsize=11, fontweight='bold', color='#333333')
    
    # Add Logo inside the bar
    if logo_path and Path(logo_path).exists():
        try:
            img = plt.imread(logo_path)
            imagebox = OffsetImage(img, zoom=0.08) 
            ab = AnnotationBbox(imagebox, (x_pos, height * 0.85), frameon=False)
            ax.add_artist(ab)
        except Exception:
            pass 

def main():
    json_files = glob.glob("results/json/*.json")
    if not json_files:
        print("Error: No JSON files found.")
        return
    
    latest_file = max(json_files, key=lambda f: Path(f).stat().st_mtime)
    with open(latest_file, "r") as f:
        res = json.load(f)
        
    step = res["step"]
    crowd_key = "Crowd-v1"
    
    data = []
    for task in TASKS:
        if task in res["results"]:
            data.append({"Model": crowd_key, "Task": task, "Score": get_primary_metric(res["results"][task]) * 100})
        else:
            data.append({"Model": crowd_key, "Task": task, "Score": 0.0})
            
        for qwen_name, qwen_scores in QWEN_BASELINES.items():
            data.append({"Model": qwen_name, "Task": task, "Score": qwen_scores[task] * 100})
            
    df = pd.DataFrame(data)
    
    # Increased figsize height slightly and added hspace later to fit titles
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    plt.rcParams['font.sans-serif'] = "Arial"
    models_order = ["Crowd-v1", "Qwen3.5-0.8B", "Qwen3-0.6B", "Qwen2.5-0.5B"]
    
    for i, task in enumerate(TASKS):
        ax = axes[i]
        task_df = df[df["Task"] == task].set_index("Model").reindex(models_order).reset_index()
        
        ax.set_facecolor("white")
        
        for j, row in task_df.iterrows():
            model_name = row["Model"]
            color = COLORS.get(model_name, "#CCCCCC")
            draw_bar(ax, j, row["Score"], color, row["Score"], LOGOS.get(model_name))

        # --- FIXED TITLE LOGIC ---
        title, subtitle = TASK_TITLES.get(task, (task.upper(), ""))
        
        # Use ax.text instead of ax.set_title to mix font weights!
        # y=1.15 places it 15% above the top of the plot
        ax.text(0, 1.15, title, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left', color='#111111')
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=10, fontweight='normal', va='bottom', ha='left', color='#666666')
        
        # Y-Axis styling
        ax.set_ylim(0, 110)
        ax.set_xlim(-0.8, 3.8)
        ax.set_xticks([])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='y', colors='#666666')
        ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=1, zorder=0)
        
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

    # Legend
    legend_patches = [
        patches.Patch(color=COLORS["Crowd-v1"], label=f"Crowd-v1"),
        patches.Patch(color=COLORS["Qwen3.5-0.8B"], label="Qwen3.5-0.8B"),
        patches.Patch(color=COLORS["Qwen3-0.6B"], label="Qwen3-0.6B"),
        patches.Patch(color=COLORS["Qwen2.5-0.5B"], label="Qwen2.5-0.5B"),
    ]
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12, 
               handlelength=1.5, handleheight=1.5)

    # Adjust layout to prevent overlap and make room for titles + legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.35)
    
    Path("results").mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Benchmarks grid saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
