#!/usr/bin/env python
# ------------------------------------------------------------------
# DeBERTa-v3-small fine-tuning on RTX-2080 Ti (11 GB)
# – header-proof, OOM-safe, detailed metrics
# ------------------------------------------------------------------
import os, re, random, time, datetime, gc, json, pathlib
import numpy as np, torch

# ── 0 · ENV PATCHES *before* any HF import ────────────────────────
_bad = re.compile(r"[^:]+:\s?.+")
for k, v in list(os.environ.items()):
    if k.upper().endswith("HEADERS") and not _bad.match(v):
        os.environ.pop(k)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ── heavy imports AFTER scrub ─────────────────────────────────────
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
)
import evaluate

log = lambda m: print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)

# ── runtime & seed ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
log(f"Device: {DEVICE.upper()} | Torch {torch.__version__}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

# ── hyper-parameters ──────────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-small"
CSV_PATH    = "data.csv"
NUM_EPOCHS  = 4
BATCH_SIZE  = 8
GRAD_ACCUM  = 4          # effective batch 32
LR          = 2e-5
NUM_WORKER  = 4

# ── data load & split ────────────────────────────────────────────
raw   = load_dataset("csv", data_files=CSV_PATH)["train"].shuffle(seed=SEED)
split = raw.train_test_split(test_size=0.2, seed=SEED)
temp  = split["test"].train_test_split(test_size=0.5, seed=SEED)
dataset = DatasetDict(train=split["train"], validation=temp["train"], test=temp["test"])
log(f"Rows → train {len(dataset['train'])} | val {len(dataset['validation'])}")

# ── label encoding ───────────────────────────────────────────────
labels   = sorted(set(dataset["train"]["label"]))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
dataset  = dataset.map(lambda x: {"labels": label2id[x["label"]]}, remove_columns=["label"])

# ── tokenizer (slow mode, trunc 256) ─────────────────────────────
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, use_fast=False)
except OSError:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

dataset = dataset.map(
    lambda b: tok(b["question"], truncation=True, max_length=256),
    batched=True, num_proc=NUM_WORKER
).remove_columns(["question"])
dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8, return_tensors="pt")

# ── model (try cache first) ──────────────────────────────────────
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True,
        num_labels=len(labels), id2label=id2label, label2id=label2id)
except EnvironmentError:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels), id2label=id2label, label2id=label2id)

model.to(DEVICE)

# ── metrics ──────────────────────────────────────────────────────
from collections import OrderedDict
metric_fns = OrderedDict(
    accuracy  = evaluate.load("accuracy"),
    precision = evaluate.load("precision"),
    recall    = evaluate.load("recall"),
    f1        = evaluate.load("f1"),
    mcc       = evaluate.load("matthews_correlation"),
)

def compute_metrics(eval_pred):
    logits = eval_pred[0] if isinstance(eval_pred, tuple) else eval_pred.predictions
    labels = eval_pred[1] if isinstance(eval_pred, tuple) else eval_pred.label_ids
    preds  = np.argmax(logits, axis=-1)
    out = {}
    for name, fn in metric_fns.items():
        if name in {"precision", "recall", "f1"}:
            out[name] = fn.compute(predictions=preds, references=labels,
                                   average="macro")[name]
        else:
            out[name] = fn.compute(predictions=preds, references=labels)[name]
    return out

# ── callback for pretty logs ─────────────────────────────────────
class PrintLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and state.is_local_process_zero and "loss" in logs:
            if "eval_loss" in logs:
                log(f"Epoch {logs['epoch']:.1f} | "
                    f"val_loss {logs['eval_loss']:.4f} "
                    f"acc {logs.get('eval_accuracy',0):.4f}")
            else:
                log(f"Step {state.global_step} | loss {logs['loss']:.4f}")
        return control

# ── TrainingArguments (version-agnostic) ─────────────────────────
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

# ── train ─────────────────────────────────────────────────────────
log("Training …")
t0 = time.time()
trainer.train()
log(f"Training done in {(time.time()-t0)/60:.1f} min")

# ── save model & tokenizer ───────────────────────────────────────
trainer.save_model("deberta-csv-final")
tok.save_pretrained("deberta-csv-final")
log("Saved best model → deberta-csv-final")

# ── self-describing metrics.json ─────────────────────────────────
def add_metric(dct, key, value, desc):
    dct[key] = {"value": float(value), "description": desc}

metrics = {}

# best-epoch validation scores
for e in trainer.state.log_history:
    if "eval_accuracy" in e:
        add_metric(metrics, "best_val_epoch", e["epoch"],
                   "Epoch that achieved the highest validation accuracy.")
        add_metric(metrics, "best_val_accuracy", e["eval_accuracy"],
                   "Accuracy on the validation split at the best epoch.")
        add_metric(metrics, "best_val_loss", e["eval_loss"],
                   "Cross-entropy loss on the validation split at the best epoch.")
        add_metric(metrics, "best_val_precision", e["eval_precision"],
                   "Macro-averaged precision on the validation split at the best epoch.")
        add_metric(metrics, "best_val_recall",   e["eval_recall"],
                   "Macro-averaged recall on the validation split at the best epoch.")
        add_metric(metrics, "best_val_f1",       e["eval_f1"],
                   "Macro-averaged F1 score on the validation split at the best epoch.")
        add_metric(metrics, "best_val_mcc",      e["eval_mcc"],
                   "Matthews correlation coefficient on the validation split at the best epoch.")
        break

# final training loss
for e in reversed(trainer.state.log_history):
    if "loss" in e and "eval_loss" not in e:
        add_metric(metrics, "final_train_loss", e["loss"],
                   "Cross-entropy loss on the last training step.")
        break

# single evaluation on test set
test_res = trainer.evaluate(dataset["test"])
add_metric(metrics, "test_accuracy",  test_res["eval_accuracy"],
           "Accuracy on the held-out test set.")
add_metric(metrics, "test_loss",      test_res["eval_loss"],
           "Cross-entropy loss on the test set.")
add_metric(metrics, "test_precision", test_res["eval_precision"],
           "Macro-averaged precision on the test set.")
add_metric(metrics, "test_recall",    test_res["eval_recall"],
           "Macro-averaged recall on the test set.")
add_metric(metrics, "test_f1",        test_res["eval_f1"],
           "Macro-averaged F1 score on the test set.")
add_metric(metrics, "test_mcc",       test_res["eval_mcc"],
           "Matthews correlation coefficient on the test set.")

path = pathlib.Path("metrics.json")
path.write_text(json.dumps(metrics, indent=2))
log(f"Metrics saved → {path.resolve()}")

# ── cleanup stats ────────────────────────────────────────────────
if DEVICE == "cuda":
    torch.cuda.empty_cache(); gc.collect()
    log(f"GPU residual {torch.cuda.memory_allocated()/1e9:.2f} GB")
try:
    import psutil, os as _os
    log(f"CPU RSS {psutil.Process(_os.getpid()).memory_info().rss/1e9:.2f} GB")
except ModuleNotFoundError:
    pass
log("✓ Finished")
