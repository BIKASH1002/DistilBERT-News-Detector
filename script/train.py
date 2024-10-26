from utils.dataset import load_dataset
from models.model import get_model
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from script.evaluation import evaluate_model, plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def model_init():
    return get_model("distilbert-base-uncased")

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_texts, val_texts, train_labels, val_labels = load_dataset('C:/Users/bikas/Desktop/Fake news detection/data/train.csv')
    
    train_encodings = tokenizer(list(train_texts), truncation = True, padding = True)
    val_encodings = tokenizer(list(val_texts), truncation = True, padding = True)
    
    train_dataset = NewsDataset(train_encodings, train_labels.tolist())
    val_dataset = NewsDataset(val_encodings, val_labels.tolist())
    
    model = get_model(model_name)
    
    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = './logs',
        evaluation_strategy = 'epoch',
    )
    
    trainer = Trainer(
        model_init = model_init,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
    )
    
    trainer.train()
    trainer.save_model('../model/fake_news_detector')
    
    evaluate_model(trainer, val_dataset)

    

if __name__ == "__main__":
    main()


