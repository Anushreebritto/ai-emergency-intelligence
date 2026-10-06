# AI Emergency Intelligence System

An NLP-based humanitarian information classification system that uses a fine-tuned DistilBERT Transformer to classify emergency and disaster-related text into 10 response-oriented categories.

## Overview

During disasters, large volumes of social media and textual reports can contain information about injuries, evacuations, infrastructure damage, urgent needs, rescue efforts, and public support.

This project applies Natural Language Processing and Transformer-based text classification to automatically categorize such information into meaningful humanitarian event types.

The system accepts emergency-related text and returns:

* Predicted humanitarian category
* Confidence score
* Top-3 predictions
* Classification status
* Model information

> This is an AI-assisted information classification prototype. It does not perform autonomous emergency response, severity prediction, or real-world dispatch decisions.

## Problem Statement

Emergency-related information is often unstructured and difficult to organize manually at scale.

The objective of this project is to build a supervised NLP classification system capable of identifying the type of humanitarian information expressed in disaster-related text.

### Research Question

Can a fine-tuned Transformer model reliably classify disaster-related text into response-oriented humanitarian information categories?

## Key Features

* Multi-class emergency information classification
* Fine-tuned DistilBERT Transformer
* Transfer learning from a pretrained language model
* 10 humanitarian event categories
* Confidence scoring using softmax probabilities
* Top-3 prediction ranking
* Held-out test-set evaluation
* Confusion matrix and error analysis
* Interactive Gradio interface
* Hugging Face Model Hub integration

## System Workflow

```text
Emergency Text Input
        ↓
Text Tokenization
        ↓
DistilBERT Transformer
        ↓
Classification Head
        ↓
Softmax Probabilities
        ↓
Predicted Category
        ↓
Confidence + Top-3 Predictions
```

## Dataset

The project uses the **HumAID Event Type Dataset** from the Qatar Computing Research Institute (QCRI), available through Hugging Face.

**Dataset:** `QCRI/HumAID-event-type`

Four disaster configurations were combined:

* Fire
* Flood
* Earthquake
* Hurricane

### Dataset Size

| Split      |    Samples |
| ---------- | ---------: |
| Training   |     53,531 |
| Validation |      7,793 |
| Test       |     15,160 |
| **Total**  | **76,484** |

## Classification Categories

The model predicts one of 10 humanitarian event types:

1. Caution and Advice
2. Displaced People and Evacuations
3. Infrastructure and Utility Damage
4. Injured or Dead People
5. Missing or Found People
6. Not Humanitarian
7. Other Relevant Information
8. Requests or Urgent Needs
9. Rescue, Volunteering or Donation Effort
10. Sympathy and Support

## Machine Learning Approach

### Base Model

**DistilBERT (`distilbert-base-uncased`)**

The project uses transfer learning rather than training a Transformer from scratch.

A pretrained DistilBERT model was fine-tuned for supervised 10-class sequence classification.

### Training Configuration

* Learning rate: `2e-5`
* Batch size: `16`
* Epochs: `2`
* Weight decay: `0.01`
* Validation performed during training
* Best model selected based on validation accuracy

### Processing Pipeline

```text
HumAID Dataset
      ↓
Dataset Combination
      ↓
Label Mapping
      ↓
DistilBERT Tokenization
      ↓
Transformer Fine-Tuning
      ↓
Validation
      ↓
Held-Out Test Evaluation
```

## Model Performance

The final model was evaluated on a held-out test set containing **15,160 examples**.

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **78.32%** |
| Macro F1    | **76.36%** |
| Weighted F1 | **77.87%** |

### Per-Class Performance

| Category                                | F1 Score |
| --------------------------------------- | -------: |
| Caution and Advice                      |   70.17% |
| Displaced People and Evacuations        |   89.68% |
| Infrastructure and Utility Damage       |   82.00% |
| Injured or Dead People                  |   93.06% |
| Missing or Found People                 |   80.58% |
| Not Humanitarian                        |   60.15% |
| Other Relevant Information              |   57.42% |
| Requests or Urgent Needs                |   59.66% |
| Rescue, Volunteering or Donation Effort |   87.67% |
| Sympathy and Support                    |   83.22% |

## Error Analysis

A confusion matrix was used to identify systematic classification errors.

The major sources of confusion were:

* `not_humanitarian` ↔ `other_relevant_information`
* `other_relevant_information` → `rescue_volunteering_or_donation_effort`
* `other_relevant_information` → `caution_and_advice`
* `requests_or_urgent_needs` → `rescue_volunteering_or_donation_effort`

The `other_relevant_information` category was the most challenging major class, with an F1 score of **57.42%**.

These errors indicate semantic overlap between broad humanitarian categories rather than purely random prediction errors.

## Example Prediction

### Input

> Urgent medical supplies are urgently needed at the evacuation center. Volunteers are requesting drinking water, food, and first aid kits.

### Model Output

**Predicted Category:** Requests or Urgent Needs

**Confidence:** 86.3%

### Top-3 Predictions

| Rank | Category                                | Confidence |
| ---- | --------------------------------------- | ---------: |
| 1    | Requests or Urgent Needs                |      86.3% |
| 2    | Rescue, Volunteering or Donation Effort |      10.2% |
| 3    | Sympathy and Support                    |       0.9% |

The top-3 output provides additional context when multiple humanitarian categories have semantic overlap.

## Technology Stack

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* Scikit-learn

### NLP

* Natural Language Processing
* Multi-class Text Classification
* Tokenization
* Transfer Learning
* Transformer Fine-Tuning
* Softmax Confidence Scoring

### Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1
* Weighted F1
* Confusion Matrix
* Error Analysis

### Application

* Gradio
* Hugging Face Model Hub

### Development

* Git
* GitHub
* VS Code

## Project Structure

```text
ai-emergency-intelligence/
│
├── app.py
├── predict.py
│
├── train_model.py
├── prepare_dataset.py
├── explore_data.py
├── evaluatemodel.py
├── confusion_matrix.py
│
├── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── README.md
```

The trained model is hosted separately on Hugging Face because the model weights are approximately 255 MB.

## Hugging Face Model

The fine-tuned DistilBERT model is hosted on Hugging Face Model Hub:

**[emergency-intelligence-distilbert](https://huggingface.co/Anushreebritto2/emergency-intelligence-distilbert)**

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Anushreebritto/ai-emergency-intelligence.git
cd ai-emergency-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The application loads the fine-tuned model from Hugging Face and launches the interactive Gradio interface.

## Limitations

* Performance depends on the language and context represented in the training data.
* Some humanitarian categories have substantial semantic overlap.
* `Other Relevant Information`, `Requests or Urgent Needs`, and `Not Humanitarian` are comparatively difficult categories.
* The system is intended for information classification and should not be treated as an autonomous emergency decision-making system.
* The model was trained and evaluated primarily on disaster-related social media text represented by the HumAID dataset.

## Future Work

Potential improvements include:

* Class-balanced training strategies
* Targeted data augmentation for weaker classes
* Threshold-based uncertainty detection
* More detailed error analysis
* Domain adaptation to additional emergency information sources
* Multilingual emergency-text classification
* Human-in-the-loop review for uncertain predictions
* Evaluation on external disaster datasets

## Author

**Anushree Britto**

B.E. Artificial Intelligence & Data Science
Prathyusha Engineering College

GitHub: **[Anushreebritto](https://github.com/Anushreebritto)**

## Disclaimer

This project is an academic and research-oriented prototype demonstrating Transformer-based humanitarian information classification.

It is not intended to replace trained emergency personnel, official disaster-management systems, or human decision-making.
