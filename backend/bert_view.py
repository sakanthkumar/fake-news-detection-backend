from termcolor import colored
import csv
from datetime import datetime

def print_menu():
    print("\n=== BERT Fake News Detector ===")
    print("1. Enter headline manually")
    print("2. Show all saved predictions")
    print("3. Export predictions to CSV")
    print("4. Search predictions by keyword")
    print("5. Exit")

def input_choice():
    return input("\nSelect an option (1-5): ")

def input_headline():
    return input("\nEnter your headline:\n")

def input_search_keyword():
    return input("\nEnter search term: ")

def show_prediction(label, confidence):
    if confidence >= 0.90:
        level = "HIGH CONFIDENCE"
    elif confidence >= 0.75:
        level = "MEDIUM CONFIDENCE"
    else:
        level = "LOW CONFIDENCE"
        label = "POTENTIALLY FAKE"

    color = "green" if label == "REAL" else ("red" if label == "FAKE" else "yellow")
    print(f"\nPrediction: {colored(label, color)}")
    print(f"Confidence: {confidence*100:.2f}% ({level})")
    return label

def print_predictions(rows):
    if not rows:
        print("\n⚠️ No predictions found.")
        return
    for row in rows:
        color = "green" if row[2] == "REAL" else ("red" if row[2] == "FAKE" else "yellow")
        print(f"\nID: {row[0]}")
        print(f"Text: {row[1]}")
        print(f"Prediction: {colored(row[2], color)}")
        print(f"Confidence: {row[3]*100:.2f}%")
        print(f"Time: {row[4]}")
        print(f"Source: {row[5]}")

def export_predictions(rows):
    if not rows:
        print("\n⚠️ No predictions to export.")
        return
    filename = f"predictions_export_{datetime.utcnow().strftime('%Y-%m-%d_%H%M')}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "confidence", "timestamp", "source"])
        writer.writerows([row[1:] for row in rows])
    print(f"\n✅ Exported {len(rows)} records to {filename}.")