import re
import joblib
from transformers import BertTokenizer, BertModel
import torch
from datetime import datetime
import os

class FakeNewsModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.clf = joblib.load("bert_classifier.pkl")
        self._init_db()

    def get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    padding=True, max_length=64)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return cls_emb.reshape(1, -1)

    def predict(self, text):
        emb = self.get_embedding(text)
        pred = self.clf.predict(emb)[0]
        proba = self.clf.predict_proba(emb)[0]
        confidence = proba[pred]
        label = "REAL" if pred == 1 else "FAKE"
        return label, confidence

    def is_news_headline(self, text):
        text = text.strip()
        if len(text) < 20 or len(text.split()) < 4 or text.endswith("?"):
            return False
        if re.search(r"\b(I|you|he|she|my|your|we|our|they|their)\b", text, re.IGNORECASE):
            return False
        profanity = ["fuck", "shit", "bitch", "gay", "asshole"]
        return not any(word in text.lower() for word in profanity)

    def save_prediction(self, text, label, confidence, source):
        conn = self._get_db_conn()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            INSERT INTO predictions (text, label, confidence, timestamp, source)
            VALUES (?, ?, ?, ?, ?)
        """, (text, label, float(confidence), timestamp, source))
        conn.commit()
        cursor.close()
        conn.close()

    def get_predictions(self, keyword=None):
        conn = self._get_db_conn()
        cursor = conn.cursor()
        if keyword:
            cursor.execute("""
                SELECT id, text, label, confidence, timestamp, source
                FROM predictions
                WHERE text LIKE ?
                ORDER BY id DESC
            """, (f"%{keyword}%",))
        else:
            cursor.execute("""
                SELECT id, text, label, confidence, timestamp, source
                FROM predictions
                ORDER BY id DESC
            """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
