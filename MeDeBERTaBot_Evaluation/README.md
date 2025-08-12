# MeDeBERTaBot_Evaluation — MedMCQA LLM‑judge Harness

This folder contains a self‑contained evaluation harness for **medical QA** on **MedMCQA**.  
Your production QA system is treated as a black box (HTTP API), while a lightweight **LLM‑judge** (MedGemma‑27B UD‑IQ2_M via `llama.cpp`) maps free‑text answers to MCQ labels (A/B/C/D) and computes **accuracy**. The harness also emits a **full per‑item protocol**: question, options, your system’s answer, judge output, gold label, correctness, and latency.

## Contents
- `eval_medmcqa.py` — evaluation script (GPU‑aware, judge auto‑download, robust label schema support, full protocol JSON, bytes→str guard).
- `requirements.txt` — minimal Python dependencies for datasets/metrics/IO and GPU judge.

---

## 1) Environment & Installation (Local)

Use Python ≥ 3.10 in a clean virtualenv/conda env.

```bash
python -m venv .env-eval
source .env-eval/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### GPU build of `llama-cpp-python` (preferred)
Install a prebuilt CUDA wheel (no `nvcc` needed). Pick the index that matches your CUDA runtime (e.g., cu122). Example:
```bash
pip install --index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
  llama-cpp-python==0.2.90
```
If you must build from source (e.g., on systems without matching wheels):
```bash
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DGGML_QKK_64=on -DCMAKE_CUDA_ARCHITECTURES=75" \
  pip install --no-binary :all: llama-cpp-python==0.2.90
```

> **Stability toggles (optional):**
> ```bash
> export GGML_CUDA_USE_GRAPHS=0
> export GGML_CUDA_FORCE_MMQ=0
> export GGML_CUDA_FORCE_DMMV=1
> ```

---

## 2) Quick Start

Evaluate first 5 items of the validation split and write a protocol JSON:
```bash
python eval_medmcqa.py \
  --split validation \
  --limit 5 \
  --json-out scores_val.json
```

**Notes**
- Most MedMCQA **test** mirrors lack gold labels → use `validation` or `train` for accuracy.
- Default judge context is `n_ctx=768`; all layers offloaded to GPU (`n_gpu_layers=-1`). On 11 GB GPUs (RTX 2080 Ti), this fits with headroom.

---

## 3) How It Works (High Level)

1) Loads MedMCQA split (robust fallbacks).  
2) Calls your QA API (`POST https://medebertabot.e-rehab.pp.ua/ask`) with JSON `{"password": "...", "text": "<question>"}` and extracts `response` (free text).  
3) The judge (MedGemma‑27B UD‑IQ2_M, GGUF via `llama.cpp`) reads the question, your answer, and the A/B/C/D options, and outputs a **single letter**.  
4) Predictions and gold labels are normalized to **0..3** and **accuracy** is computed.  
5) A **full protocol** JSON is written (per‑item details + summary + system metadata).

---

## 4) CLI Summary

```text
--judge-path /path/to/medgemma-27b-text-it-UD-IQ2_M.gguf  (auto-download if absent)
--judge-repo unsloth/medgemma-27b-text-it-GGUF
--judge-file medgemma-27b-text-it-UD-IQ2_M.gguf
--split train|validation|test
--limit N                  # evaluate first N items
--protocol-cap K           # store only first K protocol entries in JSON
--n-ctx 768                # judge context; reduce for VRAM headroom
--n-gpu-layers -1          # layers on GPU (-1 = all); reduce to free VRAM
--json-out scores.json     # write results + full protocol
```

---

## 5) Google Colab (T4) Workflow

### Install latest `llama-cpp-python` with GPU support
```python
!CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

### Test GPU availability
```python
from llama_cpp import llama_print_system_info
print(llama_print_system_info().decode("utf-8", errors="ignore"))
```

### Install other dependencies
```python
!python -m pip install -U "datasets>=2.20.0" "evaluate>=0.4.2" "huggingface_hub>=0.23.0" \
                          "requests>=2.32.0" "pyarrow>=14.0.2" "pandas>=2.2.2" "scikit-learn>=1.4.0"
```

### Running the eval script
```python
!python /content/eval_medmcqa.py \
  --split validation \
  --limit 100 \
  --json-out /content/scores_val.json
```

> Tips: If you hit OOM or rare CUDA faults, try `--n-ctx 512`, `--n-gpu-layers 70`, and set the stability env‑vars before importing `llama_cpp`:
> ```python
> import os
> os.environ["GGML_CUDA_USE_GRAPHS"] = "0"
> os.environ["GGML_CUDA_FORCE_MMQ"]   = "0"
> os.environ["GGML_CUDA_FORCE_DMMV"]  = "1"
> ```

---

## 6) Output JSON (schema excerpt)

```json
{
  "meta": {
    "timestamp_utc": "...",
    "script": "eval_medmcqa.py",
    "hf_datasets_or_evaluate_version": "...",
    "llama_cpp_system_info": "CUDA = 1, cuBLAS = 1, ..."
  },
  "config": {
    "judge_repo": "...",
    "judge_file": "...",
    "judge_path": "...",
    "split": "validation",
    "limit": 5,
    "protocol_cap": null,
    "n_ctx": 768,
    "n_gpu_layers": -1,
    "qa_endpoint": "https://medebertabot.e-rehab.pp.ua/ask",
    "request_timeout_sec": 75
  },
  "results": {
    "summary": {
      "accuracy": 0.6,
      "n_samples_total": 5,
      "n_samples_scored": 5,
      "n_samples_skipped": 0,
      "wall_time_sec": 99.76,
      "samples_per_sec": 0.0501
    },
    "protocol": [
      {
        "index": 0,
        "question": "...",
        "options": { "a": "...", "b": "...", "c": "...", "d": "..." },
        "system_answer": "...",
        "judge": {
          "chosen_letter": "b",
          "chosen_index": 1,
          "raw_output": "B",
          "latency_ms": 180.12
        },
        "gold": { "letter": "b", "index": 1 },
        "correct": true
      }
    ]
  }
}
```

---

## 7) Troubleshooting

- **No labels in split**: HF `test` often lacks gold labels → use `--split validation` or `train`.
- **CUDA illegal access / `MUL_MAT failed`**: set env toggles (see above), try `--n-ctx 512`, `--n-gpu-layers 70`, keep `n_batch=128` (default).
- **`GLIBCXX_3.4.29 not found`**: prefer the prebuilt CUDA wheel; else install newer `libstdc++.so.6` (e.g., conda‑forge `libstdcxx-ng`) and ensure it precedes system libs in `LD_LIBRARY_PATH`.
- **Bytes not JSON serializable**: handled internally (bytes→str guard for `llama_print_system_info()`).
- **Low throughput**: ensure GPU wheel is in use (system info shows `CUDA = 1`).

---

## 8) Security

The QA endpoint expects a password; **do not commit secrets**. If needed, move the password to an environment variable before publishing.

---

## 9) Extending

- **PubMedQA / Yes‑No tasks**: map free text to `{yes,no,maybe}`, compute macro‑F1.
- **Long‑form QA**: consider ROUGE/BERTScore or rubric‑based LLM graders.
- **Evidence logging**: record citations/IDs from your QA service for downstream audits.
