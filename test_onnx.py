import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

MODEL_PATH = "./optimized_model/model.onnx"
TOKENIZER_PATH = "./optimized_model"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH)

text = (
    "Urgent medical supplies are urgently needed at the evacuation center. "
    "Volunteers are requesting drinking water, food, and first aid kits."
)

inputs = tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    padding=True,
    max_length=128,
)

onnx_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
}

outputs = session.run(None, onnx_inputs)
logits = outputs[0]

probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
predicted_id = int(np.argmax(probabilities, axis=1)[0])
confidence = float(np.max(probabilities))

labels = {
    0: "caution_and_advice",
    1: "displaced_people_and_evacuations",
    2: "infrastructure_and_utility_damage",
    3: "injured_or_dead_people",
    4: "missing_or_found_people",
    5: "not_humanitarian",
    6: "other_relevant_information",
    7: "requests_or_urgent_needs",
    8: "rescue_volunteering_or_donation_effort",
    9: "sympathy_and_support",
}

print("ONNX Prediction Test")
print("--------------------")
print(f"Category: {labels[predicted_id]}")
print(f"Confidence: {confidence * 100:.2f}%")