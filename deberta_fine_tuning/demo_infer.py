#!/usr/bin/env python
# demo_infer.py – quick inference with the fine-tuned model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "deberta-csv-final"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

# fix: ensure the mapping uses int keys
id2label = {int(k): v for k, v in model.config.id2label.items()}

def predict(text: str):
    batch = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**batch).logits
    pred_id = logits.argmax(-1).item()
    prob    = logits.softmax(-1)[0, pred_id].item()
    return id2label[pred_id], prob

if __name__ == "__main__":
    while True:
        q = input("Ask> ")
        if not q:
            break
        lbl, p = predict(q)
        print(f"→ {lbl}  (p={p:.3f})")
