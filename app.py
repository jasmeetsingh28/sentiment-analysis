import streamlit as st
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Initialize NLTK and download required resources
nltk.download('punkt', quiet=True)


# Model initialization
@st.cache_resource
def load_model():
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return tokenizer, model


def preprocess_text(text):
    # Basic preprocessing
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())


def predict_sentiment(text, tokenizer, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess text
    processed_text = preprocess_text(text)

    # Tokenize
    inputs = tokenizer(
        processed_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Map prediction to sentiment
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_map[prediction.item()]
    confidence = probabilities.max().item() * 100

    return predicted_sentiment, confidence


def main():
    st.title("Sentiment Analysis App")

    # Load model
    tokenizer, model = load_model()

    # Create text input box
    user_input = st.text_area("Enter your comment:", height=100)

    # Create submit button
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Show loading spinner while processing
            with st.spinner('Analyzing sentiment...'):
                sentiment, confidence = predict_sentiment(user_input, tokenizer, model)

                # Display results
                st.write("---")
                st.write(f"**Text:** {user_input}")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Confidence:** {confidence:.2f}%")


if __name__ == "__main__":
    main()
