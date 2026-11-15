#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgb
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

LOGOS = {
    "Crowd-v1": "assets/logos/crowd.png",
    "Qwen2.5-0.5B": "assets/logos/qwen25.png",
    "Qwen3-0.6B": "assets/logos/qwen3.png",
    "Qwen3.5-0.8B": "assets/logos/qwen35.png",
}

COLORS = {
    "Crowd-v1": "#D1D5DB",      # Gray 300
    "Qwen3.5-0.8B": "#8B5CF6",  # Violet 500
    "Qwen3-0.6B": "#A78BFA",    # Violet 400
    "Qwen2.5-0.5B": "#C4B5FD"   # Violet 300
}

def get_primary_metric(metrics: dict) -> float:
    for k in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if k in metrics: return metrics[k]
    return 0.0

def draw_gradient_bar(ax, x_pos, height, width, color_hex):
    if height <= 0: return
    
    h = height
    w = width
    r_x = 0.15
    r_y = r_x * 25
    
    verts = [
        (x_pos - w/2, 0), (x_pos - w/2, h - r_y), (x_pos - w/2, h), 
        (x_pos - w/2 + r_x, h), (x_pos + w/2 - r_x, h), (x_pos + w/2, h), 
        (x_pos + w/2, h - r_y), (x_pos + w/2, 0), (x_pos - w/2, 0)
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    path = MplPath(verts, codes)
    
    patch = patches.PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    rgb = to_rgb(color_hex)
    gradient = np.zeros((100, 1, 4))
    gradient[:, :, :3] = rgb
    gradient[:, :, 3] = np.linspace(0.55, 1.0, 100).reshape(-1, 1)

    im = ax.imshow(gradient, extent=(x_pos - w/2, x_pos + w/2, 0, h),
                   aspect='auto', zorder=3, origin='lower', interpolation='bilinear')
    im.set_clip_path(patch)

def draw_bar(ax, x_pos, height, color, value, logo_path=None):
    width = 0.6  
    draw_gradient_bar(ax, x_pos, height, width, color)
    
    # Value Text
    ax.text(x_pos, height + 2, f"{value:.1f}", ha='center', va='bottom', 
            fontsize=11, fontweight='bold', color='#111111', zorder=4)
    
    if height >= 15 and logo_path and Path(logo_path).exists():
        try:
            zoom_size = 0.12 if height > 20 else 0.08 
            img = plt.imread(logo_path)
            imagebox = OffsetImage(img, zoom=zoom_size) 
            
            ab = AnnotationBbox(
                imagebox, (x_pos, height), 
                # CHANGED: -23 is the perfect middle ground, and pad=0.35 tightens the box
                xybox=(0, -23), 
                boxcoords="offset points",
                box_alignment=(0.5, 0.5),
                frameon=True, 
                bboxprops=dict(boxstyle="round,pad=0.35,rounding_size=0.2", 
                               facecolor="white", edgecolor="white", alpha=1.0),
                zorder=5
            )
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
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')
    axes = axes.flatten()
    
    plt.rcParams['font.sans-serif'] = "Arial"
    models_order = ["Crowd-v1", "Qwen3.5-0.8B", "Qwen3-0.6B", "Qwen2.5-0.5B"]
    
    for i, task in enumerate(TASKS):
        ax = axes[i]
        task_df = df[df["Task"] == task].set_index("Model").reindex(models_order).reset_index()
        ax.set_facecolor("white")
        
        for j, row in task_df.iterrows():
            model_name = row["Model"]
            draw_bar(ax, j, row["Score"], COLORS.get(model_name, "#CCCCCC"), row["Score"], LOGOS.get(model_name))

        # Titles
        title, subtitle = TASK_TITLES.get(task, (task.upper(), ""))
        ax.text(0, 1.15, title, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left', color='#111111')
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=10, fontweight='normal', va='bottom', ha='left', color='#666666')
        
        # Y-Axis styling
        ax.set_ylim(0, 115)
        ax.set_xlim(-0.8, 3.8)
        ax.set_xticks([])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='y', colors='#A0A0A0', length=0)
        
        # Faint grid lines
        ax.grid(axis='y', color='#F3F4F6', linestyle='-', linewidth=2, zorder=0)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    legend_patches = [
        patches.Patch(color=COLORS["Crowd-v1"], label="Crowd-v1"),
        patches.Patch(color=COLORS["Qwen3.5-0.8B"], label="Qwen3.5-0.8B"),
        patches.Patch(color=COLORS["Qwen3-0.6B"], label="Qwen3-0.6B"),
        patches.Patch(color=COLORS["Qwen2.5-0.5B"], label="Qwen2.5-0.5B"),
    ]
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12, 
               handlelength=1.5, handleheight=1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.35)
    
    Path("results").mkdir(exist_ok=True)
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print(f"✅ Benchmarks grid saved to {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
