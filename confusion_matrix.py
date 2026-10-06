from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. LOAD THE SAME DATASET USED FOR TRAINING
# --------------------------------------------------

configs = ["fire", "flood", "earthquake", "hurricane"]

datasets = []

for config in configs:
    ds = load_dataset("QCRI/HumAID-event-type", config)
    datasets.append(ds)


# Combine the four disaster datasets
test_dataset = concatenate_datasets(
    [ds["test"] for ds in datasets]
)

print(f"Test examples: {len(test_dataset)}")


# --------------------------------------------------
# 2. CREATE LABEL MAPPINGS
# --------------------------------------------------

all_labels = sorted(
    set(
        example["class_label"]
        for ds in datasets
        for example in ds["train"]
    )
)

label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}

print("\nLabels:")
for i, label in id2label.items():
    print(f"{i}: {label}")


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

model_path = "./emergency_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"\nUsing device: {device}")


# --------------------------------------------------
# 4. TOKENIZE TEST DATA
# --------------------------------------------------

texts = [
    str(text) if text is not None else ""
    for text in test_dataset["tweet_text"]
]

true_labels = [
    label2id[label]
    for label in test_dataset["class_label"]
]

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)


# --------------------------------------------------
# 5. RUN MODEL PREDICTIONS
# --------------------------------------------------

predictions = []

batch_size = 32

print("\nRunning predictions...")

with torch.no_grad():

    for start in range(0, len(texts), batch_size):

        end = start + batch_size

        input_ids = encodings["input_ids"][start:end].to(device)
        attention_mask = encodings["attention_mask"][start:end].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predicted = torch.argmax(
            outputs.logits,
            dim=1
        )

        predictions.extend(
            predicted.cpu().numpy()
        )


predictions = np.array(predictions)
true_labels = np.array(true_labels)


# --------------------------------------------------
# 6. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=list(range(len(all_labels)))
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# --------------------------------------------------
# 7. SAVE CONFUSION MATRIX IMAGE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=all_labels
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d"
)

plt.title("HumAID Emergency Information Classification")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved: confusion_matrix.png")


# --------------------------------------------------
# 8. CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predictions,
        target_names=all_labels,
        digits=4
    )
)


# --------------------------------------------------
# 9. TOP 10 MISCLASSIFICATION PAIRS
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 MISCLASSIFICATION PAIRS")
print("=" * 60)

misclassifications = []

for actual in range(len(all_labels)):

    for predicted in range(len(all_labels)):

        # Ignore correct predictions
        if actual != predicted:

            count = cm[actual][predicted]

            if count > 0:

                misclassifications.append(
                    (
                        count,
                        all_labels[actual],
                        all_labels[predicted]
                    )
                )


# Sort from most frequent to least frequent
misclassifications.sort(
    reverse=True,
    key=lambda x: x[0]
)


for rank, (count, actual, predicted) in enumerate(
    misclassifications[:10],
    start=1
):

    print(f"\n{rank}.")
    print(f"Actual:    {actual}")
    print(f"Predicted: {predicted}")
    print(f"Count:     {count}")


# --------------------------------------------------
# 10. OVERALL ERROR COUNT
# --------------------------------------------------

correct = np.sum(true_labels == predictions)
total = len(true_labels)
errors = total - correct

print("\n" + "=" * 60)
print("OVERALL ERROR SUMMARY")
print("=" * 60)

print(f"Total test examples : {total}")
print(f"Correct predictions : {correct}")
print(f"Incorrect predictions: {errors}")
print(f"Accuracy            : {correct / total:.4f}")