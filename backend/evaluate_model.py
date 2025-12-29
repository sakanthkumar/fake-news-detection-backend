import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate():
    print("Loading model and test data...")
    try:
        # Load model
        if not os.path.exists("bert_classifier.pkl"):
            print("[ERROR] Model 'bert_classifier.pkl' not found.")
            return

        clf = joblib.load("bert_classifier.pkl")
        
        # Load test data
        if os.path.exists("bert_testset.pkl"):
            X_test, y_test = joblib.load("bert_testset.pkl")
        else:
            print("[ERROR] Test set 'bert_testset.pkl' not found. Cannot evaluate on original test set.")
            return

        print(f"[OK] Loaded test set with {len(X_test)} samples.")

        # Predict
        print("Running predictions...")
        y_pred = clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Prepare output string
        output_lines = []
        output_lines.append("="*40)
        output_lines.append("MODEL EVALUATION REPORT")
        output_lines.append("="*40)
        output_lines.append(f"\nModel Accuracy: {acc:.2%}\n")
        output_lines.append("-" * 20)
        output_lines.append("Classification Report:")
        output_lines.append("-" * 20)
        output_lines.append(report)
        output_lines.append("-" * 20)
        output_lines.append("Confusion Matrix:")
        output_lines.append("-" * 20)
        output_lines.append(str(cm))
        
        output_text = "\n".join(output_lines)
        
        # Print to console
        print(output_text)
        
        # Save to file
        report_filename = "evaluation_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(output_text)
            
        print(f"\n[OK] Report successfully saved to '{report_filename}'")
        
    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate()
