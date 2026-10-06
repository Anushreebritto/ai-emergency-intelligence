from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "./emergency_model"

# -----------------------------------------
# 1. Load test datasets
# -----------------------------------------

configs = ["fire", "flood", "earthquake", "hurricane"]

datasets = []

for config in configs:
    print(f"Loading {config}...")
    dataset = load_dataset("QCRI/HumAID-event-type", config)
    datasets.append(dataset)

test = concatenate_datasets([d["test"] for d in datasets])

# -----------------------------------------
# 2. Label mapping
# -----------------------------------------

labels = sorted(set(test["class_label"]))

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def encode_labels(example):
    example["label"] = label2id[example["class_label"]]
    return example

test = test.map(encode_labels)

# -----------------------------------------
# 3. Load tokenizer
# -----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_function(examples):
    return tokenizer(
        examples["tweet_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test = test.map(tokenize_function, batched=True)

# -----------------------------------------
# 4. Prepare test dataset
# -----------------------------------------

columns = ["input_ids", "attention_mask", "label"]

test = test.remove_columns(
    [col for col in test.column_names if col not in columns]
)

test.set_format("torch")

# -----------------------------------------
# 5. Load trained model
# -----------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# -----------------------------------------
# 6. Create Trainer
# -----------------------------------------

trainer = Trainer(
    model=model
)

# -----------------------------------------
# 7. Make predictions
# -----------------------------------------

print("\nRUNNING FINAL TEST EVALUATION...\n")

predictions = trainer.predict(test)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = predictions.label_ids

# -----------------------------------------
# 8. Accuracy
# -----------------------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\nFINAL TEST ACCURACY:")
print(f"{accuracy:.4f}")

# -----------------------------------------
# 9. Classification report
# -----------------------------------------

print("\nCLASSIFICATION REPORT:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=labels,
        digits=4
    )
)