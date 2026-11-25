
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Model Configuration
# -----------------------------
# We will replace this with your actual Hugging Face model ID later.
MODEL_NAME = "YOUR_HUGGING_FACE_USERNAME/YOUR_MODEL_REPOSITORY"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review, tokenizer, model, device):
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)[0]

    predicted_class = torch.argmax(probabilities).item()

    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"

    return {
        "sentiment": sentiment,
        "confidence": probabilities[predicted_class].item() * 100,
        "negative": probabilities[0].item() * 100,
        "positive": probabilities[1].item() * 100
    }


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and let our DistilBERT model "
    "predict whether it expresses a positive or negative sentiment."
)

review = st.text_area(
    "Enter your movie review:",
    placeholder="Write your movie review here...",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        with st.spinner("Analyzing your review..."):
            try:
                tokenizer, model, device = load_model()
                result = predict_sentiment(review, tokenizer, model, device)

                st.subheader("Prediction")

                if result["sentiment"] == "POSITIVE":
                    st.success("POSITIVE SENTIMENT")
                else:
                    st.error("NEGATIVE SENTIMENT")

                st.metric(
                    "Confidence",
                    f'{result["confidence"]:.2f}%'
                )

                st.subheader("Class Probabilities")

                st.write(f'Positive: {result["positive"]:.2f}%')
                st.progress(result["positive"] / 100)

                st.write(f'Negative: {result["negative"]:.2f}%')
                st.progress(result["negative"] / 100)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Built with DistilBERT, PyTorch and Streamlit")
