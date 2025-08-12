#!/usr/bin/env python3
# ------------------------------------------------------------
# eval_medmcqa.py — Automatic evaluation on MedMCQA
# Judge: MedGemma-27B (UD-IQ2_M, GGUF) via llama-cpp-python
# QA endpoint: https://medebertabot.e-rehab.pp.ua/ask  (timeout=75s)
# Auto-downloads judge GGUF if missing (huggingface_hub)
# Robust gold-label extraction + letter→int mapping for metrics
# Writes a FULL testing protocol (per-item details) into JSON
# Bytes→str guard for JSON (fixes Colab 'bytes not JSON serializable')
# ------------------------------------------------------------
import os
# --- CUDA stability toggles (safe defaults; override via env if needed)
os.environ.setdefault("GGML_CUDA_USE_GRAPHS", "0")
os.environ.setdefault("GGML_CUDA_FORCE_MMQ", "0")
os.environ.setdefault("GGML_CUDA_FORCE_DMMV", "1")

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import requests
from datasets import load_dataset, disable_caching
import evaluate
from llama_cpp import Llama, llama_print_system_info
from huggingface_hub import hf_hub_download

# ----------------------------- Config -----------------------------

DEFAULT_REPO_ID = "unsloth/medgemma-27b-text-it-GGUF"
DEFAULT_FILENAME = "medgemma-27b-text-it-UD-IQ2_M.gguf"

QA_ENDPOINT = "https://medebertabot.e-rehab.pp.ua/ask"
REQUEST_TIMEOUT_SEC = 75

LETTER2INT = {"a": 0, "b": 1, "c": 2, "d": 3}
INT2LETTER = {v: k for k, v in LETTER2INT.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------- Utils ------------------------------

def _b2s(x):
    """Bytes → str (UTF-8, ignore errors); pass-through for non-bytes."""
    return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x

# ------------------------- Download helper ------------------------

def ensure_judge_file(judge_path: Optional[Path],
                      repo_id: str,
                      filename: str) -> Path:
    """
    Return a filesystem path to the GGUF judge.
    If `judge_path` exists, use it; otherwise download from Hugging Face.
    """
    if judge_path is not None and judge_path.exists():
        logging.info(f"Using local judge file: {judge_path}")
        return judge_path

    logging.info(
        f"Judge file not found locally. Downloading '{filename}' "
        f"from repo '{repo_id}' via huggingface_hub …"
    )
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logging.info(f"Downloaded/cached judge at: {cached_path}")
    return Path(cached_path)

# ---------------------- Judge initialisation ----------------------

def load_judge(gguf_path: str, n_ctx: int = 768, n_gpu_layers: int = -1, seed: int = 42) -> Llama:
    """
    Load the quantised MedGemma-27B judge via llama.cpp.
    n_gpu_layers=-1 places all layers on GPU (fits on 11 GB with UD-IQ2_M).
    """
    return Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        seed=seed,
        verbose=True,          # prints backend info (CUDA/cuBLAS) at startup
        n_batch=128,           # modest batch to reduce scratch/KV pressure
    )

def judge_option_with_raw(llm: Llama, question: str, system_answer: str, options: list[str]) -> Tuple[str, str]:
    """
    Return (chosen_letter, raw_model_text). Deterministic (temperature=0).
    """
    prompt = (
        "You are a medical board examiner.\n"
        f"Question: {question}\n"
        f"System answer: {system_answer}\n"
        + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options)) +
        "\nWhich option (A, B, C or D) matches the answer? Reply with a single letter."
    )
    out = llm(prompt, max_tokens=4, temperature=0.0)["choices"][0]["text"]
    m = re.search(r"[ABCDabcd]", out)
    letter = m.group(0).lower() if m else "a"   # safe fallback
    return letter, out

# ------------------------- QA system call -------------------------

def my_medical_qa_system(question: str) -> str:
    """
    Send the question to your HTTPS QA service and return a plain-text answer.
    Service JSON format (new):
      {"input": "<echoed question>", "response": "<free-text answer>"}
    Falls back to {"answer": "..."} or {"text": "..."} if needed.
    """
    message = {
        "password": "password",   # replace if needed
        "text": question
    }
    try:
        r = requests.post(
            QA_ENDPOINT,
            data=json.dumps(message),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        r.raise_for_status()
        payload = r.json()
        resp_text = (
            payload.get("response")
            or payload.get("answer")   # legacy fallback
            or payload.get("text")     # very conservative fallback
            or ""
        )
        return str(resp_text).strip()

    except (requests.RequestException, ValueError) as e:
        logging.warning(f"QA service error: {e!s}")
        return ""

# ------------------------- Dataset utilities ----------------------

def load_medmcqa_split(split: str):
    """
    Robust loader for MedMCQA across common HF dataset IDs.
    """
    disable_caching()
    tried = []

    try:
        tried.append("medmcqa (default)")
        ds = load_dataset("medmcqa")[split]
        return ds
    except Exception:
        pass

    try:
        tried.append("openlifescienceai/medmcqa")
        ds = load_dataset("openlifescienceai/medmcqa")[split]
        return ds
    except Exception:
        pass

    try:
        tried.append('medmcqa with config="validation"')
        ds = load_dataset("medmcqa", "validation")[split]
        return ds
    except Exception as e:
        raise RuntimeError(f"Unable to load MedMCQA split='{split}'. Tried: {tried}") from e

def extract_gold_letter(row) -> Optional[str]:
    """
    Normalize gold label to {'a','b','c','d'} from various dataset schemas.
    Supports: 'cop' (1..4 or 0..3), 'answer', 'label', 'correct', 'target'
    (ints or letters). Returns None if not found/parsable.
    """
    letters = ["a", "b", "c", "d"]

    def int_to_letter(v: int) -> Optional[str]:
        if v in (0, 1, 2, 3):     # 0-based
            return letters[v]
        if v in (1, 2, 3, 4):     # 1-based
            return letters[v - 1]
        return None

    # Canonical MedMCQA: 'cop' is usually 1..4
    if "cop" in row and row["cop"] is not None:
        v = row["cop"]
        if isinstance(v, int):
            return int_to_letter(v)
        if isinstance(v, str):
            vs = v.strip()
            if vs.isdigit():
                return int_to_letter(int(vs))
            m = re.search(r"[abcd]", vs.lower())
            if m:
                return m.group(0)

    # Alternative keys in mirrors/exports
    for key in ("answer", "label", "correct", "target"):
        if key in row and row[key] is not None:
            v = row[key]
            if isinstance(v, int):
                return int_to_letter(v)
            if isinstance(v, str):
                vs = v.strip().lower()
                m = re.search(r"[abcd]", vs)
                if m:
                    return m.group(0)
                if vs.isdigit():
                    return int_to_letter(int(vs))

    return None

def letter_to_int(x: str) -> Optional[int]:
    """Map 'a'/'b'/'c'/'d' → 0..3, else None."""
    if not isinstance(x, str):
        return None
    return LETTER2INT.get(x.lower())

# --------------------------- Evaluation ---------------------------

def run_eval(judge: Llama,
             split: str = "validation",
             limit: Optional[int] = None,
             protocol_cap: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute accuracy and return a dict with summary + full protocol.
    - protocol_cap: if set, store only first K protocol entries (size control).
    """
    ds = load_medmcqa_split(split)

    # Guard: require labels
    cols = set(ds.column_names)
    if not any(k in cols for k in ("cop", "answer", "label", "correct", "target")):
        raise RuntimeError(
            f"Split '{split}' has no gold labels (cop/answer/label/correct/target). "
            "Choose --split validation or --split train."
        )

    if limit:
        ds = ds.select(range(limit))

    metric = evaluate.load("accuracy")
    preds_int, gold_int = [], []
    protocol: list[Dict[str, Any]] = []
    skipped = 0

    t0 = time.perf_counter()
    for i, row in enumerate(ds):
        q = row["question"]
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]

        # Call your system
        sys_ans = my_medical_qa_system(q)

        # Judge with latency timing
        jt0 = time.perf_counter()
        choice_letter, judge_raw = judge_option_with_raw(judge, q, sys_ans, options)
        judge_ms = (time.perf_counter() - jt0) * 1000.0

        gold_letter = extract_gold_letter(row)
        ci = letter_to_int(choice_letter)
        gi = letter_to_int(gold_letter) if gold_letter is not None else None

        item: Dict[str, Any] = {
            "index": int(i),
            "question": str(q),
            "options": {
                "a": str(options[0]),
                "b": str(options[1]),
                "c": str(options[2]),
                "d": str(options[3]),
            },
            "system_answer": str(sys_ans),
            "judge": {
                "chosen_letter": str(choice_letter),
                "chosen_index": None if ci is None else int(ci),
                "raw_output": str(judge_raw).strip(),
                "latency_ms": round(judge_ms, 2),
            },
            "gold": None,
            "correct": None,
            "skipped": False,
            "skip_reason": None,
        }

        if gi is not None:
            item["gold"] = {"letter": str(gold_letter), "index": int(gi)}

        if ci is None or gi is None:
            skipped += 1
            item["skipped"] = True
            item["skip_reason"] = f"Unmappable labels (pred={choice_letter!r}, gold={gold_letter!r})"
        else:
            correct = (ci == gi)
            item["correct"] = bool(correct)
            preds_int.append(ci)
            gold_int.append(gi)

        protocol.append(item)

        if (i + 1) % 100 == 0:
            logging.info(f"Progress: {i+1}/{len(ds)}")

    if len(gold_int) == 0:
        raise RuntimeError("No valid gold labels found; cannot compute accuracy.")

    score = metric.compute(predictions=preds_int, references=gold_int)
    elapsed = time.perf_counter() - t0

    stored_protocol = protocol if (protocol_cap is None) else protocol[:protocol_cap]

    result = {
        "summary": {
            "accuracy": float(score["accuracy"]),
            "n_samples_total": int(len(ds)),
            "n_samples_scored": int(len(gold_int)),
            "n_samples_skipped": int(skipped),
            "wall_time_sec": float(elapsed),
            "samples_per_sec": float(len(gold_int) / elapsed if elapsed > 0 else 0.0),
        },
        "protocol": stored_protocol,
    }
    return result

# ------------------------------- CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a QA system on MedMCQA using MedGemma-27B (UD-IQ2_M, GGUF) as an LLM judge and save a full protocol."
    )
    ap.add_argument("--judge-path", type=Path, default=None,
                    help="Path to local GGUF file. If missing, it will be downloaded.")
    ap.add_argument("--judge-repo", default=DEFAULT_REPO_ID,
                    help="Hugging Face repo ID (default: unsloth/medgemma-27b-text-it-GGUF).")
    ap.add_argument("--judge-file", default=DEFAULT_FILENAME,
                    help="Filename within the repo if local file absent.")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, help="Evaluate only first N examples")
    ap.add_argument("--protocol-cap", type=int, default=None,
                    help="Store only first K items of the full protocol in JSON (size control).")
    ap.add_argument("--n-ctx", type=int, default=768, help="Judge context window (tokens)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="Layers on GPU (-1 = all). Reduce to free VRAM.")
    ap.add_argument("--json-out", type=Path, help="Write results + full protocol to JSON")
    args = ap.parse_args()

    start_ts = datetime.utcnow().isoformat() + "Z"

    judge_file_path = ensure_judge_file(args.judge_path, args.judge_repo, args.judge_file)

    logging.info("Loading judge LLM …")
    judge = load_judge(str(judge_file_path), n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)

    logging.info(f"Running evaluation on MedMCQA/{args.split} …")
    res = run_eval(judge, split=args.split, limit=args.limit, protocol_cap=args.protocol_cap)

    # Compose final JSON payload with metadata/config
    payload = {
        "meta": {
            "timestamp_utc": start_ts,
            "script": "eval_medmcqa.py",
            "hf_datasets_or_evaluate_version": str(evaluate.__version__),
            "llama_cpp_system_info": _b2s(llama_print_system_info()),
        },
        "config": {
            "judge_repo": str(args.judge_repo),
            "judge_file": str(args.judge_file),
            "judge_path": str(judge_file_path),
            "split": str(args.split),
            "limit": None if args.limit is None else int(args.limit),
            "protocol_cap": None if args.protocol_cap is None else int(args.protocol_cap),
            "n_ctx": int(args.n_ctx),
            "n_gpu_layers": int(args.n_gpu_layers),
            "qa_endpoint": QA_ENDPOINT,
            "request_timeout_sec": int(REQUEST_TIMEOUT_SEC),
            # Record CUDA stability envs actually used:
            "env": {
                "GGML_CUDA_USE_GRAPHS": os.getenv("GGML_CUDA_USE_GRAPHS"),
                "GGML_CUDA_FORCE_MMQ": os.getenv("GGML_CUDA_FORCE_MMQ"),
                "GGML_CUDA_FORCE_DMMV": os.getenv("GGML_CUDA_FORCE_DMMV"),
            },
        },
        "results": res,
    }

    # Print summary to console
    print("\n=== RESULTS (summary) ===")
    for k, v in res["summary"].items():
        print(f"{k:>20}: {v}")

    # Save full protocol if requested
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full protocol → {args.json_out}")

if __name__ == "__main__":
    main()
