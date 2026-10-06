from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "Anushreebritto2/emergency-intelligence-distilbert"
# Load trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_emergency(text):
    # Convert text into model inputs
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)

    # Get highest probability
    predicted_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_id].item()

    # Convert ID back to category name
    predicted_label = model.config.id2label[predicted_id]

    return predicted_label, confidence


if __name__ == "__main__":

    text = input("\nEnter emergency information: ")

    label, confidence = predict_emergency(text)

    print("\nPREDICTION")
    print("----------")
    print("Category:", label)
    print(f"Confidence: {confidence * 100:.2f}%")