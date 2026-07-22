# 🛍️ Amazon Review NLP Dashboard

An end-to-end Natural Language Processing (NLP) project that analyzes Amazon product reviews using machine learning, transformer models, semantic clustering, and large language models (LLMs).

The project includes an interactive Streamlit dashboard where users can explore sentiment predictions, discover review clusters, and compare different summarization approaches.

---

# 📖 Overview

Customer reviews contain valuable insights about products, but manually reading thousands of reviews is time-consuming.

This project demonstrates how modern NLP techniques can be combined to automatically:

- Classify review sentiment
- Group semantically similar reviews
- Summarize customer feedback
- Extract common themes and insights

The application was built as an end-to-end NLP pipeline, from data preprocessing to an interactive visualization dashboard.

---

# 📂 Dataset

The project uses Amazon product reviews containing:

- Review title
- Review text
- Rating
- Sentiment label

After preprocessing, the reviews are used throughout the NLP pipeline.

---

# 🚀 Features

## 😊 Sentiment Analysis

Compare predictions from two different sentiment analysis models:

### TF-IDF + Logistic Regression

- Traditional machine learning approach
- Fast and lightweight
- Uses TF-IDF features

### RoBERTa Transformer

- Pretrained transformer model
- Context-aware predictions
- Better understanding of natural language

---

## 🧩 Review Clustering

Reviews are grouped according to semantic similarity using:

- SentenceTransformers embeddings
- K-Means clustering
- PCA visualization

Each cluster includes:

- Top keywords
- Sentiment distribution
- Example reviews

---

## 📝 Review Summarization

Two summarization techniques are compared.

### BART

Transformer-based abstractive summarization.

### OpenAI GPT

Structured summaries including:

- Themes
- Positive aspects
- Negative aspects
- Overall sentiment
- Natural language summary

---

# 🔄 NLP Pipeline

```text
Raw Reviews
      │
      ▼
Preprocessing
      │
      ▼
Sentiment Classification
(Logistic Regression & RoBERTa)
      │
      ▼
Sentence Embeddings
      │
      ▼
K-Means Clustering
      │
      ▼
Review Summarization
(BART & OpenAI)
      │
      ▼
Interactive Streamlit Dashboard
```

---

# 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- SentenceTransformers
- Hugging Face Transformers
- OpenAI API
- Streamlit
- Matplotlib
- Plotly

---

# 📁 Project Structure

```text
.
├── app/
│   ├── app.py
│   └── pages/
│       ├── 1_Sentiment_Analysis.py
│       ├── 2_Review_Clustering.py
│       └── 3_Summarization.py
│
├── data/
│
├── outputs/
│   ├── clustered_reviews.csv
│   ├── classifier.joblib
│   ├── vectorizer.joblib
│   ├── cluster_keywords.json
│   └── summaries/
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_classifier.py
│   ├── generate_clusters.py
│   └── generate_summaries.py
│
└── src/
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/amazon-review-nlp.git

cd amazon-review-nlp
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file containing your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

---

# ▶️ Running the Project

Run the preprocessing pipeline:

```bash
python scripts/preprocess_data.py
```

Train the sentiment classifier:

```bash
python scripts/train_classifier.py
```

Generate review clusters:

```bash
python scripts/generate_clusters.py
```

Generate summaries:

```bash
python scripts/generate_summaries.py
```

Launch the dashboard:

```bash
streamlit run app/app.py
```

---

# 📊 Dashboard

The Streamlit application contains three pages:

### 😊 Sentiment Analysis

- Compare Logistic Regression and RoBERTa predictions
- Interactive sentiment prediction

### 🧩 Review Clustering

- PCA visualization
- Cluster exploration
- Keywords
- Sentiment distribution
- Example reviews

### 📝 Summarization

Compare:

- BART summaries
- OpenAI structured summaries

---

# 📈 Results

This project demonstrates how different NLP techniques complement one another:

- Classical machine learning provides fast sentiment classification.
- Transformer models improve contextual understanding.
- Sentence embeddings enable semantic clustering.
- Large language models generate structured insights from customer feedback.

Together, these techniques create an end-to-end workflow for extracting meaningful information from large collections of text.

---

# 🔮 Future Improvements

Possible extensions include:

- Fine-tuning transformer models on Amazon reviews
- BERTopic for topic modeling
- Interactive filtering by product category
- Deployment to Streamlit Community Cloud
- Automated evaluation of generated summaries
- Real-time review ingestion
