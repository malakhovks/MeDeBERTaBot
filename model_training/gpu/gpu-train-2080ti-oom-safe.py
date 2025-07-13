#!/usr/bin/env python
# ------------------------------------------------------------------
# DeBERTa fine-tune on RTX-2080 Ti (11 GB) – OOM-safe + header-proof
# ------------------------------------------------------------------
import os, re, random, time, datetime, gc
import numpy as np, torch

# ── 0 · ENV PATCHES BEFORE _any_ HF/HUB IMPORT ─────────────────────
# 0-a. nuke **any** malformed headers env-var
_hdr_pat = re.compile(r"[^:]+:\s?.+")
for k, v in list(os.environ.items()):
    if k.upper().endswith("HEADERS") and not _hdr_pat.match(v):
        os.environ.pop(k)

# 0-b. allocator & telemetry
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ------------------------------------------------------------------
# heavy imports AFTER the scrub
# ------------------------------------------------------------------
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
)
import evaluate

log = lambda m: print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)

# ───────── runtime & seed ──────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
log(f"Device: {DEVICE.upper()} | Torch {torch.__version__}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

# ───────── hyper-params ───────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-small"
CSV_PATH    = "data.csv"
NUM_EPOCHS  = 4
BATCH_SIZE  = 8          # step fits 11 GB
GRAD_ACCUM  = 4          # effective batch 32
LR          = 2e-5
NUM_WORKER  = 4

# ───────── data load & split ──────────────────────────────────────
raw = load_dataset("csv", data_files=CSV_PATH)["train"].shuffle(seed=SEED)
split = raw.train_test_split(test_size=0.2, seed=SEED)
temp  = split["test"].train_test_split(test_size=0.5, seed=SEED)
dataset = DatasetDict(train=split["train"], validation=temp["train"], test=temp["test"])
log(f"Rows → train {len(dataset['train'])} | val {len(dataset['validation'])}")

# ───────── label encoding ─────────────────────────────────────────
labels = sorted(set(dataset["train"]["label"]))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
dataset = dataset.map(lambda x: {"labels": label2id[x["label"]]}, remove_columns=["label"])

# ───────── tokeniser (truncate 256) ───────────────────────────────
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, use_fast=False)
except OSError:
    # not cached yet → download once
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = dataset.map(
    lambda b: tok(b["question"], truncation=True, max_length=256),
    batched=True, num_proc=NUM_WORKER
).remove_columns(["question"])
dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8, return_tensors="pt")

# ───────── model (try cache first) ────────────────────────────────
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True,
        num_labels=len(labels), id2label=id2label, label2id=label2id)
except EnvironmentError:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels), id2label=id2label, label2id=label2id)

model.to(DEVICE)

# ───────── metric & callback ──────────────────────────────────────
accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    logits = p[0] if isinstance(p, tuple) else p.predictions
    labels = p[1] if isinstance(p, tuple) else p.label_ids
    return accuracy.compute(predictions=np.argmax(logits, -1), references=labels)

class PrintLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # ← add **kwargs
        if logs and state.is_local_process_zero and "loss" in logs:
            if "eval_loss" in logs:
                log(f"Epoch {logs['epoch']:.1f} | "
                    f"val_loss {logs['eval_loss']:.4f} "
                    f"acc {logs.get('eval_accuracy', 0):.4f}")
            else:
                log(f"Step {state.global_step} | loss {logs['loss']:.4f}")
        return control      # make sure to return the control object

# ───────── TrainingArguments (version-agnostic keys) ──────────────
ta = dict(
    output_dir                 = "deberta-csv-exp",
    per_device_train_batch_size= BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    gradient_accumulation_steps= GRAD_ACCUM,
    dataloader_num_workers     = NUM_WORKER,
    dataloader_pin_memory      = (DEVICE=="cuda"),
    num_train_epochs           = NUM_EPOCHS,
    learning_rate              = LR,
    seed                       = SEED,
    fp16                       = True,
    gradient_checkpointing     = False,
    optim                      = "adamw_torch_fused",
    load_best_model_at_end     = True,
    metric_for_best_model      = "accuracy",
    logging_strategy           = "steps",
    logging_steps              = 50,
    report_to                  = [],
)
if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
    ta.update(eval_strategy="epoch", save_strategy="epoch")
else:
    ta.update(evaluation_strategy="epoch", save_strategy="epoch")

args = TrainingArguments(**ta)

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["validation"],
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  callbacks=[PrintLoss])

# ───────── train ──────────────────────────────────────────────────
log("Training …")
t0 = time.time()
trainer.train()
log(f"Training done in {(time.time()-t0)/60:.1f} min")

# ───────── save & cleanup ─────────────────────────────────────────
trainer.save_model("deberta-csv-final"); tok.save_pretrained("deberta-csv-final")
log("Saved best model → deberta-csv-final")

# ---------- NEW  block: dump metrics to a JSON file ---------------
import json, pathlib
metrics = {}

# 1) grab the epoch with the highest validation accuracy
best_eval = None
for e in trainer.state.log_history:
    if "eval_accuracy" in e:
        if best_eval is None or e["eval_accuracy"] > best_eval.get("eval_accuracy", -1):
            best_eval = e
if best_eval:
    metrics["best_val_epoch"]    = best_eval["epoch"]
    metrics["best_val_accuracy"] = best_eval["eval_accuracy"]
    metrics["best_val_loss"]     = best_eval["eval_loss"]

# 2) final training loss (last step)
for entry in reversed(trainer.state.log_history):
    if "loss" in entry and "epoch" in entry:
        metrics["final_train_loss"] = entry["loss"]
        break

# 3) run one test-set evaluation
test_result = trainer.evaluate(dataset["test"])
metrics["test_accuracy"] = test_result.get("eval_accuracy")
metrics["test_loss"]     = test_result.get("eval_loss")

# 4) write to disk
metrics_path = pathlib.Path("metrics.json")
metrics_path.write_text(json.dumps(metrics, indent=2))
log(f"Metrics saved → {metrics_path.resolve()}")
# -----------------------------------------------------------------

if DEVICE == "cuda":
    torch.cuda.empty_cache(); gc.collect()
    log(f"GPU residual {torch.cuda.memory_allocated()/1e9:.2f} GB")
try:
    import psutil, os as _os
    log(f"CPU RSS {psutil.Process(_os.getpid()).memory_info().rss/1e9:.2f} GB")
except ModuleNotFoundError:
    pass
log("✓ Finished")