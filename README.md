# CrowdGPT evaluation toolkit

## Installation
```bash
git clone https://github.com/Vxtzq/CrowdEval && cd CrowdEval
pip install -r requirements.txt
```
## Singular benchmark (to push on the main repo to be seen on site)
```bash
python eval_checkpoint.py --benchmark
```
This benchmarks the latest Crowd-v1 checkpoint on **piqa**, **winogrande**, **arc_easy**, **mmlu**, **hellaswag** and **lambada-openai**
## Harder benchmarks (GSM8K and GPQA)
```bash
python harder_bench.py
```

then

```bash
python plot_hard.py
```

then

```bash
python plot_benchmark.py
```
This benchmarks the latest Crowd-v1 checkpoint on **gsm8k** and **gpqa**

## Crowd-v1 full base-model evaluation

The evaluator reads the step-specific `model_bf16.safetensors` and
`engram_bf16.safetensors` files from the Hugging Face Git history, then runs
`lm-evaluation-harness` on GPU. It deliberately ignores the static
`model.safetensors` file. The base model is scored by continuation likelihood;
no chat template or generation prompt is applied.

Tasks: HellaSwag, PIQA, ARC-Easy, WinoGrande, LAMBADA OpenAI, and MMLU. Each
task uses the harness task's default few-shot setting (MMLU is 5-shot). The
evaluator uses batch size 4. The harness right-pads causal batches, and the
model's causal mask prevents those trailing pads from affecting scored tokens.

Step 280 predates the tokenizer files; its evaluation uses the tokenizer files
from step 300, which have identical Git blob IDs at steps 300, 320, 340, and 360.
The Mistral regex compatibility fix is enabled for all checkpoints.

Run the currently selected interval checkpoints with:

```bash
source /venv/main/bin/activate
python /workspace/crowd-eval/run_milestones.py \
  --repo-dir /workspace/.crowd-hf-history-full \
  --start-step 280 --end-step 360 \
  --tokenizer-dir /workspace/crowd-eval/tokenizer
```

Per-step results are written as JSON under `results/`. Create a compact table
and a step-by-step score chart with:

```bash
python /workspace/crowd-eval/summarize_results.py /workspace/crowd-eval/results
```

This writes `summary.csv` and `score_by_step.png`.

The evaluator pins model code to the step-360 implementation so checkpoint
comparisons use the same forward code, including older checkpoints whose
repository config predates the Transformers metadata.
