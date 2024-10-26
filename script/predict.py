import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.dataset import load_test_data
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, paddings = True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim = 1)

if __name__ == "__main__":
    model_path = '../model/fake_news_detector'
    model, tokenizer = load_model(model_path)

    test_data_path = '../data/test.csv'
    test_texts = load_test_data(test_data_path)

    predictions = [predict(text, model, tokenizer) for text in test_texts]

    results = pd.DataFrame({'text': test_texts, 'prediction': predictions})
    results['label'] = results['prediction'].apply(lambda x: 'Fake News' if x == 1 else 'Real News')
    results.to_csv('../data/test_predictions.csv', index=False)
    print("Predictions saved to '../data/test_predictions.csv'")
