from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_PATH = "Anushreebritto2/emergency-intelligence-distilbert"

print("Loading ONNX model from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    subfolder="optimized_model"
)

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    subfolder="optimized_model"
)

print("ONNX model loaded successfully!")

text = (
    "Urgent medical supplies are urgently needed at the evacuation center. "
    "Volunteers are requesting drinking water, food, and first aid kits."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128,
)

outputs = model(**inputs)
logits = outputs.logits

predicted_id = int(logits.argmax(dim=-1)[0])
confidence = float(logits.softmax(dim=-1)[0][predicted_id])

print(f"Predicted ID: {predicted_id}")
print(f"Category: {model.config.id2label[predicted_id]}")
print(f"Confidence: {confidence * 100:.2f}%")