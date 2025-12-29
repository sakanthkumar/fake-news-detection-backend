import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import joblib

df = pd.read_csv("data/Mega_Combined_With_Extra_Synthetic.csv")
print("✅ Loaded dataset:", len(df))

texts = df["text"].astype(str).fillna("").tolist()
labels = df["label"].values

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.eval()

device = torch.device("cpu")
model.to(device)

BATCH_SIZE = 16

def get_embeddings_batch(text_batch):
    with torch.no_grad():
        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

all_embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeddings = get_embeddings_batch(batch_texts)
    all_embeddings.append(batch_embeddings)

embeddings = np.vstack(all_embeddings)
print("✅ Embeddings shape:", embeddings.shape)

joblib.dump(embeddings, "bert_embeddings.npy")
joblib.dump(labels, "bert_labels.npy")
print("✅ Saved bert_embeddings.npy and bert_labels.npy")