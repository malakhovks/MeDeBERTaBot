#!/usr/bin/env python
# evaluate_finetuned.py – sweep over test split

import json, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk            # we saved cached arrow files
from tqdm import tqdm

MODEL_DIR = "deberta-csv-final"
DATA_DIR  = "deberta-cached-test"              # optional: save test arrow
BATCH     = 64

tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda").eval()
id2label = {int(k):v for k,v in model.config.id2label.items()}
label2id = {v:k for k,v in id2label.items()}

# load the same test split you used during training
from datasets import load_dataset
test_ds = load_dataset("csv", data_files="data.csv")["train"]\
            .shuffle(seed=42).train_test_split(test_size=0.2, seed=42)\
            ["test"].train_test_split(test_size=0.5, seed=42)["test"]

y_true, y_pred, y_prob = [], [], []

for i in tqdm(range(0, len(test_ds), BATCH), desc="Evaluating"):
    batch = test_ds[i:i+BATCH]["question"]
    toks  = tok(batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        out = model(**toks).logits
    probs = out.softmax(-1)
    y_prob.append(probs.cpu().numpy())
    y_pred.extend(probs.argmax(-1).cpu().tolist())
    y_true.extend([label2id[lbl] for lbl in test_ds[i:i+BATCH]["label"]])

y_prob = np.vstack(y_prob)

# 2-A  text report -----------------------------------------------------------
report = classification_report(y_true, y_pred, target_names=id2label.values(), digits=3)
print(report)

# 2-B  confusion matrix ------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=id2label.values(), yticklabels=id2label.values())
plt.title("Confusion matrix – fine-tuned DeBERTa")
plt.tight_layout()
plt.savefig("confusion_matrix.png"); plt.close()

# 2-C  macro ROC-AUC ---------------------------------------------------------
auc = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
print(f"Macro ROC-AUC: {auc:.4f}")

# optional: save one micro-avg ROC curve plot
RocCurveDisplay.from_predictions(
    y_true, y_prob, name="macro-avg", plot_chance_level=True)
plt.savefig("roc_curve.png"); plt.close()

# 2-D  append to metrics.json -----------------------------------------------
mfile = pathlib.Path("metrics.json")
metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
metrics["classification_report"] = report
metrics["macro_roc_auc"]         = {"value": float(auc),
                                    "description": "Macro-averaged ROC-AUC on test set."}
mfile.write_text(json.dumps(metrics, indent=2))
print("Updated metrics.json ← confusion_matrix.png, roc_curve.png")
