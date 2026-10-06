from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

MODEL_NAME = "distilbert-base-uncased"

# -----------------------------------------
# 1. Load disaster datasets
# -----------------------------------------

configs = ["fire", "flood", "earthquake", "hurricane"]
datasets = []

for config in configs:
    print(f"Loading {config}...")
    dataset = load_dataset("QCRI/HumAID-event-type", config)
    datasets.append(dataset)

train = concatenate_datasets([d["train"] for d in datasets])
dev = concatenate_datasets([d["dev"] for d in datasets])
test = concatenate_datasets([d["test"] for d in datasets])

# -----------------------------------------
# 2. Create label mapping
# -----------------------------------------

labels = sorted(set(train["class_label"]))

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def encode_labels(example):
    example["label"] = label2id[example["class_label"]]
    return example

train = train.map(encode_labels)
dev = dev.map(encode_labels)
test = test.map(encode_labels)

# -----------------------------------------
# 3. Tokenization
# -----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["tweet_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train = train.map(tokenize_function, batched=True)
dev = dev.map(tokenize_function, batched=True)
test = test.map(tokenize_function, batched=True)

# -----------------------------------------
# 4. Keep model inputs
# -----------------------------------------

columns = ["input_ids", "attention_mask", "label"]

train = train.remove_columns(
    [col for col in train.column_names if col not in columns]
)

dev = dev.remove_columns(
    [col for col in dev.column_names if col not in columns]
)

test = test.remove_columns(
    [col for col in test.column_names if col not in columns]
)

train.set_format("torch")
dev.set_format("torch")
test.set_format("torch")

# -----------------------------------------
# 5. Load DistilBERT
# -----------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------------------
# 6. Evaluation metric
# -----------------------------------------

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(
        predictions=predictions,
        references=labels_true
    )

# -----------------------------------------
# 7. Training configuration
# -----------------------------------------

training_args = TrainingArguments(
    output_dir="./emergency_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    report_to="none"
)

# -----------------------------------------
# 8. Trainer
# -----------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_metrics
)

# -----------------------------------------
# 9. Train
# -----------------------------------------

print("\nSTARTING TRAINING...\n")

trainer.train()

# -----------------------------------------
# 10. Save model
# -----------------------------------------

trainer.save_model("./emergency_model")
tokenizer.save_pretrained("./emergency_model")

print("\nTRAINING COMPLETE!")
print("Model saved to: ./emergency_model")