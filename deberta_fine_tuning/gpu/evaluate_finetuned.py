
"""
evaluate_finetuned.py
────────────────────────────────────────────────────────────────────
Deep evaluation for the fine-tuned DeBERTa model
• accuracy, balanced-accuracy, macro/micro/weighted P/R/F1, ROC-AUC, AP
• per-class report saved to CSV/JSON
• raw & normalised confusion matrices
• micro-average ROC and Precision-Recall curves (multiclass-safe)
"""

import argparse, json, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ─── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",   default="deberta-csv-final")
parser.add_argument("--data_csv",    default="MeDeBERTaData_Q_Small.csv")
parser.add_argument("--cached_test", default="deberta-cached-test")
parser.add_argument("-b", "--batch", type=int, default=64,
                    help="Eval batch size")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device → {device}")

# ─── Model & tokenizer ────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

#  numeric-order id2label → target_names
id2label = {int(k): v for k, v in model.config.id2label.items()}
label2id = {v: k for k, v in id2label.items()}
target_names = [id2label[i] for i in range(len(id2label))]

# ─── Test split (cached on first run) ──────────────────────────────
cache = pathlib.Path(args.cached_test)
if cache.exists():
    test_ds = load_from_disk(cache)
else:
    ds = load_dataset("csv", data_files=args.data_csv)["train"].shuffle(seed=42)
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"] \
               .train_test_split(test_size=0.5, seed=42)["test"]
    test_ds.save_to_disk(cache)
print(f"Eval rows → {len(test_ds)}")

# ─── Inference loop ───────────────────────────────────────────────
y_true, y_pred, y_prob = [], [], []

for i in tqdm(range(0, len(test_ds), args.batch), desc="Infer"):
    batch = test_ds[i:i + args.batch]
    toks = tok(batch["question"], padding=True, truncation=True,
               max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**toks).logits
    probs = logits.softmax(-1).cpu().numpy()

    y_prob.append(probs)
    y_pred.extend(probs.argmax(-1))
    y_true.extend([label2id[lbl] for lbl in batch["label"]])

y_prob = np.vstack(y_prob)  # (N, C)
print("✓ Inference done")

# ─── Global metrics ───────────────────────────────────────────────
metrics = {
    "accuracy":           accuracy_score(y_true, y_pred),
    "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
    "macro_precision":    precision_score(y_true, y_pred, average="macro",  zero_division=0),
    "macro_recall":       recall_score(   y_true, y_pred, average="macro",  zero_division=0),
    "macro_f1":           f1_score(       y_true, y_pred, average="macro",  zero_division=0),
    "weighted_f1":        f1_score(       y_true, y_pred, average="weighted", zero_division=0),
    "micro_f1":           f1_score(       y_true, y_pred, average="micro",  zero_division=0),
}

# ─── Micro-avg ROC / PR curves ────────────────────────────────────
y_bin = label_binarize(y_true, classes=np.arange(len(target_names)))  # (N, C)

# ROC
fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
roc_auc     = auc(fpr, tpr)
metrics["auc_micro"] = roc_auc

plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Multiclass ROC – micro average"); plt.legend()
plt.tight_layout(); plt.savefig("roc_curve_micro.png"); plt.close()

# PR
prec, rec, _  = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
ap_micro      = average_precision_score(y_bin, y_prob, average="micro")
metrics["ap_micro"] = ap_micro

plt.plot(rec, prec, lw=2, label=f"AP = {ap_micro:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("PR curve – micro average"); plt.legend()
plt.tight_layout(); plt.savefig("pr_curve_micro.png"); plt.close()

# ─── Per-class report → CSV/JSON ──────────────────────────────────
report = classification_report(
    y_true, y_pred, target_names=target_names,
    digits=3, output_dict=True, zero_division=0)

pd.DataFrame(report).T.to_csv("classification_report.csv", float_format="%.4f")
with open("classification_report.json", "w") as fp:
    json.dump(report, fp, indent=2)

# ─── Confusion matrices ───────────────────────────────────────────
def plot_cm(cm, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)

cm_raw  = confusion_matrix(y_true, y_pred)
cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

plot_cm(cm_raw,  "Confusion matrix (counts)", "confusion_matrix.png")
plot_cm(cm_norm, "Confusion matrix (normalised)", "confusion_matrix_norm.png")

# ─── Merge with existing metrics.json (if any) ────────────────────
mfile = pathlib.Path("metrics.json")
all_metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
all_metrics.update(metrics)
mfile.write_text(json.dumps(all_metrics, indent=2))

print("✓ Finished → metrics.json, classification_report.(csv|json), "
      "confusion matrices, roc_curve_micro.png, pr_curve_micro.png")
