![banner](https://github.com/user-attachments/assets/a26a4380-907f-4cd7-acf5-390ec658947d)

# Fake News Detector: DistilBERT-Based News Classification

<div align = "justify">

This project focuses on developing an AI-powered Fake News Detector using DistilBERT, a lightweight transformer model. With the increasing spread of misinformation, identifying fake news has become crucial. This solution fine-tunes a pre-trained language model on a dataset of real and fake news articles to classify news accurately. By leveraging tools like Hugging Face's `transformers` library and Weights & Biases (`WandB`) for tracking experiments, the project offers a transparent and reproducible workflow. The end result is a high-performance model that achieves 99.76% accuracy, making it highly effective in distinguishing real from fake news.

## Content

- [Overview](#overview)

- [Setup](#setup)

- [Features](#features)

- [Dataset](#dataset)

- [Procedure](#procedure)

- [Training Configuration](#training-configuration)

- [Essential Components](#essential-components)

- [Results](#results)

- [Visualization with WandB](#visualization-with-wandb)

- [Conclusion](#conclusion)

## Overview

This project showcases the development of a Fake News Detector using DistilBERT, a lightweight LLM model. The model is trained to classify news articles as either Real or Fake, achieving high accuracy and excellent performance metrics. This repository also demonstrates the use of Hugging Face’s transformers library and Weights & Biases (WandB) for experiment tracking, ensuring a reproducible and well-documented workflow.

## Setup

- **Google Colab:** for development and training acceleration via T4 GPU

- **DistilBERT:** for training

- **WandB:** for tracking training and analysis

## Features

- **Binary Text Classification:** Identify whether a news article is real or fake.
  
- **DistilBERT Fine-tuning:** Lightweight and efficient version of BERT used for training.

- **Experiment Tracking with WandB:** Real-time logging of metrics and model performance.

- **Visualizations:** Confusion matrix, classification report, and loss trends plotted to analyze model behavior.

## Dataset

The dataset can be accessed from Kaggle. It consist of following files: 

- **Training Data:** Labeled dataset containing news articles with corresponding real or fake labels.

- **Test Data:** Unlabeled dataset used for generating predictions.

**NOTE:-** The dataset in the repository is just the chunk of the dataset. For complete access of dataset [click here](https://www.kaggle.com/competitions/smm-hw2-fakenewsdetecion/data) 

## Procedure

**1. Dataset Preparation:**

The training dataset containing labeled articles (with text and label columns) was loaded and cleaned to remove any missing values. The data was split into training and validation sets to evaluate the model's performance during training.

**2. Tokenization:**

Each article in the dataset was tokenized using DistilBERT’s tokenizer, which converts the text into input tensors that the transformer model can process. The tokenized inputs were padded and truncated to ensure uniform length across batches.

**3. Model Initialization:**

A DistilBERT-based sequence classification model was initialized with two output labels, representing real and fake news. This pretrained model was loaded from the Hugging Face library, allowing fine-tuning on the specific task of news classification.

**4. Training Configuration:**

Training arguments were configured to specify the number of epochs (3), batch sizes, learning rate warmup steps and weight decay to prevent overfitting. The model was also set to log performance metrics after each epoch.

**5. Training the Model:**

The model was trained using the Hugging Face Trainer API, which automates the process of batching, optimization, and logging. Both training and validation losses were monitored during the training phase to track convergence and ensure that the model was not overfitting.

**6. Experiment Tracking with WandB:**

The training run was integrated with Weights & Biases (WandB) to log metrics in real time. Metrics like loss, accuracy, and runtime were visualized on the WandB dashboard, helping monitor the performance of the training process.

**7. Performance Evaluation:**

After training, the model was evaluated on the validation dataset. Accuracy, precision, recall, and F1-score were calculated for both real and fake news categories. A confusion matrix was generated to visualize the distribution of correct and incorrect predictions.

**8. Generating Predictions on Test Data:**

The test dataset, which contains only the text column, was tokenized in the same way as the training data. The trained model was used to predict labels for the test data, with results saved in a CSV file.

**9. Visualization and Analysis:**

The confusion matrix, along with the trend of training and validation losses over the epochs, was plotted. These visualizations helped confirm the model's stability and effectiveness in classifying news articles accurately.

## Training Configuration

```python
training_args = TrainingArguments(
    output_dir = './results',                 # Directory to save model checkpoints
    num_train_epochs = 3,                     # Number of training epochs
    per_device_train_batch_size = 16,         # Batch size for training
    per_device_eval_batch_size = 16,          # Batch size for evaluation
    warmup_steps = 500,                       # Number of warmup steps
    weight_decay = 0.01,                      # Weight decay to prevent overfitting
    logging_dir = './logs',                   # Directory to save logs
    evaluation_strategy = 'epoch'             # Evaluate the model at the end of each epoch
)
```

## Essential Components

There are files required for the Hugging Face Transformer model such as the tokenizer configuration, model configuration, vocabulary, and tokenizer JSON. These files are essential components of the AutoTokenizer and AutoModelForSequenceClassification models used in your code. These files are fetched and stored in cache for further processing.

Here’s a brief explanation of what each of these files does:

- **tokenizer_config.json:**

Contains settings related to the tokenizer (e.g., normalization, tokenization options).

- **config.json:**

Stores the configuration for the pre-trained model, such as the number of layers, number of attention heads, and label mapping.

- **vocab.txt:**

The vocabulary file containing all possible tokens (words, subwords, or characters) that the tokenizer uses for encoding text.

- **tokenizer.json:**

Contains a detailed representation of the tokenizer logic, including token-to-ID mappings and special tokens (like `[CLS]`, `[SEP]`).

The progress can be shown as follows when the model is trained:

<div align = "center">
    <img src = "https://github.com/user-attachments/assets/6bbe90d7-4887-4699-97bb-514c91c4d7d4" alt = "Confusion matrix" width = 50%>
</div>

## Results
  
Below is the classification report generated after evaluating the model on the validation dataset:

<div align = "center">

| Metric      | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Real        | 1.00      | 1.00   | 1.00     | 2079    |
| Fake        | 1.00      | 1.00   | 1.00     | 2074    |
| **Accuracy**| -         | -      | **99.76%**| 4153   |
| Macro Avg   | 1.00      | 1.00   | 1.00     | 4153    |
| Weighted Avg| 1.00      | 1.00   | 1.00     | 4153    |

</div>

The confusion matrix generated during validation, which shows the number of correctly and incorrectly predicted instances is as follows:

<div align = "center">
    <img src = "https://github.com/user-attachments/assets/09c6e175-458d-4e41-88cd-69385772029d" alt = "Confusion matrix" width = 50%>
</div>

The training and validation loss over 3 epochs are as follows:

<div align = "center">
  
| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 0.0317        | 0.0188          |
| 2     | 0.0139        | 0.0113          |
| 3     | 0.0007        | 0.0126          |

</div>

## Visualization with WandB

During training, key metrics such as runtime, loss, and steps per second were logged using Weights & Biases (WandB). Below are key insights from the WandB visualizations:

**1. Loss Trend:** Validation loss decreased consistently, indicating convergence.

**2. Steps per Second:** Measured across epochs to monitor efficiency.

**3. Runtime Analysis:** Helps identify any bottlenecks during training.

The visualization are as follows:

![weights and bias](https://github.com/user-attachments/assets/7a59de46-1de8-40e4-b154-fa04a0195c0f)

## Conclusion

This project demonstrates a powerful and efficient approach to solving the fake news detection problem using `DistilBERT`. The model achieves a 99.76% accuracy on the validation set, with near-perfect precision, recall, and F1-score across both classes. With tools like `WandB`, the project provides a reproducible and transparent workflow, making it easy to track and improve model performance over time.

This model can be further improved by:

- Fine-tuning on larger datasets.

- Implementing hyperparameter optimization using WandB sweeps.

- Deploying the model as an API for real-time fake news detection.

</div>

## Credits

- **Kaggle:** for fake news dataset

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
