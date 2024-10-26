# Fake News Detector: DistilBERT-Based News Classification

## Content

## Overview

This project showcases the development of a Fake News Detector using DistilBERT, a lightweight LLM model. The model is trained to classify news articles as either Real or Fake, achieving high accuracy and excellent performance metrics. This repository also demonstrates the use of Hugging Face’s transformers library and Weights & Biases (WandB) for experiment tracking, ensuring a reproducible and well-documented workflow.

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

## Performance Metrics

Below is the classification report generated after evaluating the model on the validation dataset:

| Metric      | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Real        | 1.00      | 1.00   | 1.00     | 2079    |
| Fake        | 1.00      | 1.00   | 1.00     | 2074    |
| **Accuracy**| -         | -      | **99.76%**| 4153   |
| Macro Avg   | 1.00      | 1.00   | 1.00     | 4153    |
| Weighted Avg| 1.00      | 1.00   | 1.00     | 4153    |

The confusion matrix generated during validation, which shows the number of correctly and incorrectly predicted instances is as follows:

![confusion matrix](https://github.com/user-attachments/assets/09c6e175-458d-4e41-88cd-69385772029d)

The training and validation loss over 3 epochs are as follows:

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 0.0317        | 0.0188          |
| 2     | 0.0139        | 0.0113          |
| 3     | 0.0007        | 0.0126          |

