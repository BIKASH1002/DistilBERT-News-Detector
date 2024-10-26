import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap='Blues', xticklabels = ["Real", "Fake"], yticklabels = ["Real", "Fake"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(trainer, val_dataset):
    print("Evaluating on the validation dataset...")
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis = 1)
    true_labels = val_dataset.labels

    accuracy = accuracy_score(true_labels, preds)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=["Real", "Fake"]))

    cm = confusion_matrix(true_labels, preds)
    plot_confusion_matrix(cm)

