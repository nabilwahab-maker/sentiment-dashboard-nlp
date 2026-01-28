import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")
st.write(
    "This application analyzes text sentiment using a pre-trained "
    "transformer-based NLP model and visualizes the results in a dashboard."
)

# ---------------------------------
# File Upload Section
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (must contain a column named 'text')",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df = pd.read_csv("data.csv")
    st.info("Using default dataset (data.csv).")

# ---------------------------------
# Validate data
# ---------------------------------
if "text" not in df.columns:
    st.error("CSV file must contain a 'text' column.")
    st.stop()

# ---------------------------------
# Load NLP model (cached)
# ---------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_model = load_model()

# ---------------------------------
# Apply sentiment analysis
# ---------------------------------
df["raw_label"] = df["text"].apply(
    lambda x: sentiment_model(x)[0]["label"]
)

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}
df["sentiment"] = df["raw_label"].map(label_map)

score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
df["sentiment_score"] = df["sentiment"].map(score_map)

sentiment_counts = df["sentiment"].value_counts()

# ---------------------------------
# KPI Section
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Positive", sentiment_counts.get("Positive", 0))
col2.metric("Neutral", sentiment_counts.get("Neutral", 0))
col3.metric("Negative", sentiment_counts.get("Negative", 0))

st.divider()

# ---------------------------------
# Charts Section
# ---------------------------------
col4, col5 = st.columns(2)

with col4:
    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with col5:
    st.subheader("Sentiment Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["sentiment_score"], marker="o")
    ax2.set_xlabel("Text Index")
    ax2.set_ylabel("Sentiment Score")
    st.pyplot(fig2)

st.divider()

# ---------------------------------
# Data Preview
# ---------------------------------
st.subheader("Text and Sentiment Preview")

preview_df = df[["no", "text", "sentiment"]].head(10).copy()
preview_df.index = range(1, len(preview_df) + 1)

st.dataframe(preview_df)



