# 🧠 Sentiment Analysis AI

An end-to-end **Natural Language Processing (NLP)** project that predicts the sentiment of user-provided text using a custom-trained **Bidirectional LSTM (BiLSTM)** neural network. The model is deployed as a real-time interactive web application using **Streamlit**.

---

## 🚀 Overview

This project demonstrates the complete machine learning workflow, from data preprocessing and model training to deployment as a web application.

The BiLSTM model has been optimized to improve performance through:

- Bidirectional LSTM architecture
- L2 Regularization
- Spatial Dropout & Recurrent Dropout
- Learning Rate Scheduling
- Sequence Masking
- Early Stopping
- Handling Class Imbalance

The application allows users to enter text and instantly receive sentiment predictions along with confidence scores.

---

## 📂 Project Structure

```text
SENTIMENT_ANALYSIS/
│
├── dataset/                         # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── twitter_training.csv
│
├── Notebooks/                       # Model training and experiments
│   ├── training_notebook.ipynb
│   ├── test_cuda.py
│   ├── training_curves.png
│   └── confusion_matrix_dl.png
│
├── web-app/
│   ├── models/
│   │   ├── bilstm_sentiment_model.keras
│   │   ├── tokenizer.pkl
│   │   └── label_encoder.pkl
│   │
│   └── app.py
│
├── .gitignore
└── README.md
```

---

## ✨ Features

- 🔍 Real-Time Sentiment Prediction
- 😊 Predicts **Positive**, **Negative**, and **Neutral** sentiments
- 📊 Displays prediction confidence scores
- 🧠 Deep Learning model using Bidirectional LSTM
- ⚡ Fast inference with TensorFlow/Keras
- 🎨 Interactive web interface built with Streamlit
- 🛡️ Robust model with dropout and regularization
- 📈 Optimized for better generalization and reduced overfitting

---

## 🏗️ Model Architecture

The model consists of:

- Text Tokenization
- Sequence Padding
- Embedding Layer
- Bidirectional LSTM
- Dense Layers
- Dropout Layers
- Softmax Output Layer

Training techniques include:

- Early Stopping
- Learning Rate Reduction
- L2 Regularization
- Spatial Dropout
- Recurrent Dropout

---

## 📊 Model Outputs

The application predicts:

- Positive 😊
- Neutral 😐
- Negative 😞

Additionally, it displays:

- Prediction Confidence
- Probability Distribution across all classes

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### NLP & Data Processing

- Scikit-learn
- Pandas
- NumPy
- Regular Expressions (Regex)

### Web Framework

- Streamlit

### Model Serialization

- Pickle

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GaganYadav20/sentiment-analysis.git

cd sentiment-analysis
```

---

### 2. Create Virtual Environment (Optional)

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install tensorflow streamlit numpy pandas scikit-learn
```

---

### 4. Run the Application

```bash
cd web-app

streamlit run app.py
```

---

### WSL Users

If you're using **Windows Subsystem for Linux (WSL)**, run:

```bash
streamlit run app.py --server.headless true
```

---

## ⚠️ Model Path Configuration

Since the trained assets are stored inside the **models/** directory, ensure your `load_assets()` function in `app.py` points to the correct paths.

```python
@st.cache_resource
def load_assets():
    model = load_model("models/bilstm_sentiment_model.keras")

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le
```

---

## 📈 Future Improvements

- Transformer-based models (BERT, RoBERTa)
- Attention Mechanism
- Emotion Detection
- Aspect-Based Sentiment Analysis
- Multilingual Sentiment Analysis
- REST API with FastAPI
- Docker Deployment
- Hugging Face Integration

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

## ⭐ Support

If you found this project helpful:

- ⭐ Star this repository
- 🍴 Fork it
- 🐛 Report issues
- 💡 Suggest new features

---

## 👨‍💻 Author

**Gagan Yadav**

- 🎓 B.Tech CSE (AI & ML)
- 💻 Machine Learning & Web Development Enthusiast
- 🚀 Passionate about AI, NLP, Deep Learning, and Full-Stack Development

---

### 🌟 If you like this project, don't forget to give it a ⭐ on GitHub!