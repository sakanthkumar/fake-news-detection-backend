import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, BertModel
import torch

# Load resources
print("Loading model and resources...")
clf = joblib.load("bert_classifier.pkl")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
bert_model.eval()
device = torch.device("cpu")
bert_model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_emb.reshape(1, -1)

def predict_proba_texts(texts):
    # Batch transform for LIME
    # Simple loop for reproduction script (not optimized but fine for small N)
    embs = []
    for t in texts:
        embs.append(get_embedding(t)[0])
    embs = np.array(embs)
    return clf.predict_proba(embs)

headline = "virat kohli has been spotted dead yesterday"
print(f"\nAnalyzing: '{headline}'")

# 1. Raw Prediction
emb = get_embedding(headline)
probs = clf.predict_proba(emb)[0]
pred_idx = np.argmax(probs)
label = clf.classes_[pred_idx]
conf = probs[pred_idx]
print(f"Prediction: {label} (Confidence: {conf:.4f})")
print(f"Probabilities: {probs}")

# 2. LIME Explanation
print("\nRunning LIME...")
explainer = LimeTextExplainer(class_names=clf.classes_)
exp = explainer.explain_instance(headline, predict_proba_texts, num_features=6)
print("Explanation (Features contributing to prediction):")
for feature, weight in exp.as_list():
    print(f"  {feature}: {weight:.4f}")
