import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load embeddings and labels
embeddings = joblib.load("bert_embeddings.npy")
labels = joblib.load("bert_labels.npy")
print("✅ Loaded embeddings:", embeddings.shape)
print("✅ Loaded labels:", labels.shape)

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"✅ Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# Train classifier
clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.2%}\n")
print(classification_report(y_test, y_pred))

# Save classifier and test set (optional)
joblib.dump(clf, "bert_classifier.pkl")
joblib.dump((X_test, y_test), "bert_testset.pkl")
print("✅ Saved bert_classifier.pkl and bert_testset.pkl")