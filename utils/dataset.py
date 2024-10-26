import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    data = pd.read_csv(path)
    data = data.dropna(subset = ['text'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['label'], test_size = 0.2, random_state = 42)
    return train_texts, val_texts, train_labels, val_labels

def load_test_data(path):
    test_data = pd.read_csv(path)
    return test_data['text']
