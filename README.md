# Argus: Argument Understanding System

This repository contains the code and data used for our paper:

Argus: An Argument Understanding System, Leveraging Autoregressive Language Models

The project fine-tunes and evaluates autoregressive language models on argumentative discourse unit (ADU) extraction and relation identification datasets.

## Project Structure

- `train.py` — LoRA fine-tuning with Unsloth + TRL (response-only training)
- `inference.py` — Generate predictions from trained checkpoints
- `test_results_adus.py` — Evaluate ADU predictions against gold annotations
- `error_analysis.py` — Diagnose ADU-level mismatches between gold and predictions
- `datasets/` — Dataset roots (aae2, argmicro, and oracle variants)
- `output/` — Saved model checkpoints (per epoch)
- `preds/` — Inference outputs (text files per sample)

## Setup

We recommend using `uv` (fast Python package manager) or `pip`.

### Using uv

```zsh
# Sync dependencies from pyproject.toml / uv.lock
uv sync

# Activate the environment (if using uv venv)
source .venv/bin/activate
```

### Using pip

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # if you keep one, else install packages manually
```

Key Python packages: `unsloth`, `transformers`, `trl`, `datasets`, `torch`, `tqdm`, `seqeval`, `scikit-learn`.

## Training

`train_unsloth.py` fine-tunes supported models (Gemma, Llama, Qwen, Phi, TinyLlama, etc.) using Unsloth’s `train_on_responses_only()` masking.

Example (Gemma-3):
```zsh
python train_unsloth.py \
  --dataset aae2/adus \
  --model_name unsloth/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 8 \
  --gradient_steps 1 \
  --lora_r 16 \
  --remove_system_message
```

Important flags:
- `--dataset`: dataset subfolder under `datasets/`
- `--model_name`: base model (or `output/...` to resume with `--load_pretrained`)
- `--epochs`, `--batch_size`, `--gradient_steps`: training schedule
- `--lora_r`: LoRA rank
- `--remove_system_message`: merges system + user content for models that don’t support system role

Outputs are saved under `output/{model_name-normalized}-lora-r{r}-bs{...}-ac{...}-e{...}-{dataset}`.

## Inference

`inference_unsloth.py` loads a trained checkpoint and writes per-sample predictions to `preds/...`.

Single checkpoint:
```zsh
python inference_unsloth.py \
  --dataset aae2/adus \
  --model_name unsloth/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 8 \
  --gradient_steps 1 \
  --lora_r 16 \
  --remove_system_message
```

All epoch snapshots:
```zsh
python inference_unsloth.py \
  --dataset aae2/adus \
  --model_name unsloth/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 8 \
  --gradient_steps 1 \
  --lora_r 16 \
  --all_snapshots \
  --remove_system_message
```

Notes:
- The script handles Gemma-3 content format and skips prompt tokens when decoding, writing only assistant responses.
- Predictions are saved to `preds/{model-config}-{dataset}/sample-name.txt`.

## Evaluation (ADUs)

`test_results_adus.py` evaluates ADU extraction performance.

Example:
```zsh
python test_results_adus.py \
  --dataset aae2/adus \
  --model_name unsloth/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 8 \
  --gradient_steps 1 \
  --lora_r 16
```

The script reports:
- Sequence-level (BIO) metrics using `seqeval` (default and strict IOB2)
- Token-level metrics with/without the `O` label for macro F1 (for comparison)

## Error Analysis (ADU-Level)

`error_analysis.py` surfaces ADU-level mismatches between gold annotations and predictions:
- False Negative: ADU present in gold but missing in predictions
- False Positive: ADU predicted but not present in gold
- Wrong Label: Same ADU text with a different predicted label

Run:
```zsh
python error_analysis.py \
  --dataset aae2/adus \
  --model_name unsloth/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 8 \
  --gradient_steps 1 \
  --lora_r 16 \
  --max_errors 30
```

## Reproducibility

- We recommend committing `uv.lock` for applications to ensure consistent environments across machines and CI.
- Training and inference scripts generate deterministic output directories based on configuration for easier tracking.

## Notes

- Datasets are expected under `datasets/{dataset}/train` and `datasets/{dataset}/test` with JSON files containing chat-formatted messages.
- Some models don’t support the `system` role; use `--remove_system_message` to merge system + user content.
- GPU memory usage is logged during inference.

## Citation

If you use this repository or its results, please cite the Argus paper:

Argus: An Argument Understanding System, Leveraging Autoregressive Language Models
