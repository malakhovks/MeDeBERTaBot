#!/usr/bin/env python
# demo_infer.py – one-shot inference with the fine-tuned model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

MODEL_DIR = "deberta-csv-final"          # the folder we just saved
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer + model
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# grab the label map we stored in config.json
id2label = model.config.id2label

def predict(question: str):
    batch = tok(question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**batch).logits
    pred_id = logits.argmax(dim=-1).item()
    score   = logits.softmax(-1)[0, pred_id].item()
    return id2label[str(pred_id)], score          # label string + prob

if __name__ == "__main__":
    while True:
        q = input("Ask> ")
        if not q:
            break
        lbl, prob = predict(q)
        print(f"→ {lbl}  (p={prob:.3f})")
