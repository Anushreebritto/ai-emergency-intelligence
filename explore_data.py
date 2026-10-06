from datasets import load_dataset
from collections import Counter

dataset = load_dataset("QCRI/HumAID-event-type", "fire")

print("DATASET:")
print(dataset)

print("\nCOLUMNS:")
print(dataset["train"].column_names)

print("\nFIRST EXAMPLE:")
print(dataset["train"][0])

print("\nLABELS:")
print(set(dataset["train"]["class_label"]))

print("\nLABEL DISTRIBUTION:")
print(Counter(dataset["train"]["class_label"]))