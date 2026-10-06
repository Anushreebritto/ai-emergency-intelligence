from datasets import load_dataset, concatenate_datasets

configs = ["fire", "flood", "earthquake", "hurricane"]

datasets = []

for config in configs:
    print(f"Loading {config}...")
    dataset = load_dataset("QCRI/HumAID-event-type", config)
    datasets.append(dataset)

train = concatenate_datasets([d["train"] for d in datasets])
dev = concatenate_datasets([d["dev"] for d in datasets])
test = concatenate_datasets([d["test"] for d in datasets])

# Create label mappings
labels = sorted(set(train["class_label"]))

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print("\nLABEL MAPPING:")

for label, idx in label2id.items():
    print(idx, "->", label)

# Convert text labels to integers
def encode_labels(example):
    example["label"] = label2id[example["class_label"]]
    return example

train = train.map(encode_labels)
dev = dev.map(encode_labels)
test = test.map(encode_labels)

print("\nFIRST ENCODED EXAMPLE:")
print(train[0])

print("\nDATASET SIZES:")
print("Train:", len(train))
print("Dev:", len(dev))
print("Test:", len(test))