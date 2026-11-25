
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Configuration
MODEL_DIR = "."
MAX_LEN = 256

# 2. Select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# 3. Load saved tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

print("Model loaded successfully!")

# 4. Prediction function
@torch.inference_mode()
def predict_sentiment(review):

    if not review or not review.strip():
        return {}, "Please enter a movie review."

    # Tokenize review
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Forward propagation
    outputs = model(**inputs)

    # Convert logits into probabilities
    probabilities = torch.softmax(
        outputs.logits, dim=1
    )[0]

    negative = probabilities[0].item()
    positive = probabilities[1].item()

    prediction = torch.argmax(probabilities).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[prediction].item() * 100

    result = {
        "Negative": negative,
        "Positive": positive
    }

    message = f"{sentiment} Sentiment | Confidence: {confidence:.2f}%"

    return result, message


# 5. Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your movie review here...",
        label="Movie Review"
    ),
    outputs=[
        gr.Label(num_top_classes=2, label="Sentiment Probabilities"),
        gr.Textbox(label="Final Prediction")
    ],
    title="DistilBERT IMDb Sentiment Analysis",
    description="Enter a movie review to predict whether its sentiment is Positive or Negative.",
    examples=[
        ["This movie was absolutely amazing. The acting and story were excellent."],
        ["The movie was boring and a complete waste of time."],
        ["This movie was not bad at all. I actually enjoyed it."],
        ["The acting was good but the story was disappointing."]
    ]
)

# 6. Launch application
if __name__ == "__main__":
    demo.launch()
