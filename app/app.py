import streamlit as st

st.set_page_config(
    page_title="Amazon Review NLP",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Amazon Review NLP Dashboard")

st.markdown(
    """
Welcome to an end-to-end Natural Language Processing (NLP) dashboard for analyzing Amazon product reviews.

This project combines classical machine learning, transformer models, semantic embeddings, clustering, and LLM-based summarization to extract meaningful insights from customer feedback.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("😊 Sentiment Analysis")
    st.write(
        """
- TF-IDF + Logistic Regression
- RoBERTa Transformer
- Compare predictions from both models
"""
    )

with col2:
    st.subheader("🧩 Review Clustering")
    st.write(
        """
- SentenceTransformer embeddings
- K-Means clustering
- PCA visualization
- Cluster exploration
"""
    )

with col3:
    st.subheader("📝 Review Summarization")
    st.write(
        """
- BART summarization
- OpenAI structured summaries
- Themes & sentiment extraction
"""
    )

st.divider()

st.subheader("🔄 NLP Pipeline")

st.code(
    """
Raw Reviews
      ↓
Preprocessing
      ↓
Sentiment Classification
      ↓
Sentence Embeddings
      ↓
K-Means Clustering
      ↓
Review Summarization
""",
    language="text",
)

st.subheader("🛠️ Technologies")

st.markdown(
    """
- **Python**
- **Pandas**
- **Scikit-learn**
- **SentenceTransformers**
- **Hugging Face Transformers**
- **OpenAI API**
- **Streamlit**
"""
)

st.info("👈 Use the navigation menu on the left to explore each feature.")