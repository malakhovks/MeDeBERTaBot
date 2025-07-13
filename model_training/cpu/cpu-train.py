#!/usr/bin/env python
# ------------------------------------------------------------------
# DeBERTa-v3-small fine-tune on 190-core Intel CPU via oneAPI-IPEX
# OOM-safe · resume-friendly · rich metrics · NaN guard
# ------------------------------------------------------------------
import os, re, random, time, datetime, gc, json, pathlib
import numpy as np, torch
try:                                   # Intel Extension for PyTorch
    import intel_extension_for_pytorch as ipex
    IPEX = True
except ModuleNotFoundError:
    IPEX = False

# ── 0 · ENV PATCHES BEFORE _any_ HF import ────────────────────────
for k, v in list(os.environ.items()):
    if k.upper().endswith("HEADERS") and not re.match(r"[^:]+:\s?.+", v):
        os.environ.pop(k)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# fully exploit 190 logical cores
torch.set_num_threads(190)
torch.set_num_interop_threads(8)
os.environ["OMP_NUM_THREADS"] = "190"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"

log = lambda m: print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)

# ── runtime & seed ────────────────────────────────────────────────
DEVICE = "cpu"               # force CPU
log(f"Device: CPU | Torch {torch.__version__}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── hyper-parameters ─────────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-small"
CSV_PATH    = "data.csv"
NUM_EPOCHS  = 4
BATCH_SIZE  = 8
GRAD_ACCUM  = 4               # effective batch 32
LR          = 2e-5
NUM_WORKER  = 16              # optimal for 190-core box

# ── heavy imports AFTER the scrub ─────────────────────────────────
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
)
import evaluate

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

# ── tokenizer (slow mode) ────────────────────────────────────────
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

# ── model weights ────────────────────────────────────────────────
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True,
        num_labels=len(labels), id2label=id2label, label2id=label2id)
except EnvironmentError:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels), id2label=id2label, label2id=label2id)

if IPEX:
    try:
        # First, try the fast BF16 path ─ works only on AVX-512 / AMX CPUs
        model = ipex.optimize(model, dtype=torch.bfloat16)
        BF16 = True
        log("✓ IPEX BF16 path enabled")
    except AssertionError as e:
        log(f"⚠️  {e}\n   Falling back to FP32 IPEX path.")
        # Disable weight-prepack to avoid the same check
        model = ipex.optimize(model,
                              dtype=torch.float32,
                              weights_prepack=False)
        BF16 = False
else:
    BF16 = False

model.to(DEVICE)

# ── metrics setup ────────────────────────────────────────────────
from collections import OrderedDict
metric_fns = OrderedDict(
    accuracy  = evaluate.load("accuracy"),
    precision = evaluate.load("precision"),
    recall    = evaluate.load("recall"),
    f1        = evaluate.load("f1"),
    mcc       = evaluate.load("matthews_correlation"),
)
def compute_metrics(epred):
    logits = epred[0] if isinstance(epred, tuple) else epred.predictions
    labels = epred[1] if isinstance(epred, tuple) else epred.label_ids
    preds  = np.argmax(logits, axis=-1)
    d = {}
    for n, fn in metric_fns.items():
        if n in {"precision", "recall", "f1"}:
            d[n] = fn.compute(predictions=preds, references=labels, average="macro")[n]
        else:
            d[n] = fn.compute(predictions=preds, references=labels)[n]
    return d

# ── callbacks ────────────────────────────────────────────────────
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
class NanGuard(TrainerCallback):
    """Abort training if NaN/Inf appears in loss, grads, or params."""
    def on_step_end(self, args, state, control, **kw):
        model  = kw["model"]; loss = kw.get("loss")
        if loss is not None and (torch.isnan(loss) | torch.isinf(loss)).any():
            raise ValueError(f"NaN/Inf in loss @ step {state.global_step}")
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                raise ValueError(f"NaN/Inf in grads @ step {state.global_step}")
            if torch.isnan(p).any() or torch.isinf(p).any():
                raise ValueError(f"NaN/Inf in params @ step {state.global_step}")
        return control

# ── TrainingArguments ────────────────────────────────────────────
ta = dict(
    output_dir                 = "deberta-csv-exp",
    per_device_train_batch_size= BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    gradient_accumulation_steps= GRAD_ACCUM,
    dataloader_num_workers     = NUM_WORKER,
    dataloader_pin_memory      = False,
    num_train_epochs           = NUM_EPOCHS,
    learning_rate              = LR,
    seed                       = SEED,
    fp16                       = False,
    bf16                       = BF16,
    gradient_checkpointing     = False,
    optim                      = "adamw_torch",
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
                  callbacks=[PrintLoss, NanGuard])

# ── resume logic (weights-only) ──────────────────────────────────
ckpt_dir, latest = "deberta-csv-exp", None
if os.path.isdir(ckpt_dir):
    ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if ckpts:
        latest = os.path.join(ckpt_dir, sorted(ckpts, key=lambda s:int(s.split('-')[-1]))[-1])
        log(f"Resuming from {latest}")
        for fn in ("optimizer.pt", "scheduler.pt", "scaler.pt"):
            fp = os.path.join(latest, fn)
            if os.path.isfile(fp):
                os.remove(fp); log(f"Removed stale {fn}")

# ── train ─────────────────────────────────────────────────────────
log("Training …"); t0 = time.time()
trainer.train(resume_from_checkpoint=latest)
log(f"Training finished in {(time.time()-t0)/60:.1f} min")

# ── save model & tokenizer ───────────────────────────────────────
trainer.save_model("deberta-csv-final"); tok.save_pretrained("deberta-csv-final")
log("Saved best model → deberta-csv-final")

# ── write metrics.json ───────────────────────────────────────────
def put(d,k,v,desc): d[k] = {"value":float(v),"description":desc}
metrics = {}
for e in trainer.state.log_history:
    if "eval_accuracy" in e:
        put(metrics,"best_val_epoch",    e["epoch"],        "Best epoch.")
        put(metrics,"best_val_accuracy", e["eval_accuracy"],"Acc@best epoch.")
        put(metrics,"best_val_loss",     e["eval_loss"],    "Loss@best epoch.")
        put(metrics,"best_val_precision",e["eval_precision"],"Precision@best.")
        put(metrics,"best_val_recall",   e["eval_recall"],  "Recall@best.")
        put(metrics,"best_val_f1",       e["eval_f1"],      "F1@best.")
        put(metrics,"best_val_mcc",      e["eval_mcc"],     "MCC@best.")
        break
for e in reversed(trainer.state.log_history):
    if "loss" in e and "eval_loss" not in e:
        put(metrics,"final_train_loss",e["loss"],"Last-step train loss.")
        break
test = trainer.evaluate(dataset["test"])
put(metrics,"test_accuracy", test["eval_accuracy"], "Acc@test.")
put(metrics,"test_loss",     test["eval_loss"],     "Loss@test.")
put(metrics,"test_precision",test["eval_precision"],"Precision@test.")
put(metrics,"test_recall",   test["eval_recall"],   "Recall@test.")
put(metrics,"test_f1",       test["eval_f1"],       "F1@test.")
put(metrics,"test_mcc",      test["eval_mcc"],      "MCC@test.")
path = pathlib.Path("metrics.json"); path.write_text(json.dumps(metrics,indent=2))
log(f"Metrics saved → {path.resolve()}")

# ── cleanup stats ────────────────────────────────────────────────
gc.collect()
try:
    import psutil, os as _os
    log(f"CPU RSS {psutil.Process(_os.getpid()).memory_info().rss/1e9:.2f} GB")
except ModuleNotFoundError:
    pass
log("✓ Finished")
