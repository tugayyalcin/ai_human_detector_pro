# 🕵️‍♂️ AI vs Human Text Detector Pro

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=for-the-badge&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-GloVe%20%7C%20HAN-00C7B7?style=for-the-badge)

An advanced, highly optimized Natural Language Processing (NLP) pipeline designed to accurately classify text as either **Human-written** or **AI-generated**. 

This project implements a **Hierarchical Attention Network (HAN)** supercharged with **Stanford's GloVe Pre-trained Word Vectors**, achieving state-of-the-art accuracy while maintaining a lightweight hardware footprint optimized for both Apple Silicon (M4) and NVIDIA GPUs (RTX 3050).

---

## 📁 Project Structure & Files Overview

The repository is structured for modularity and scalability. Here is the anatomy of the project:

* **`src/preprocessing.py`**: The cleaning engine. It handles text normalization, lowercasing, punctuation removal, and tokenization to prepare raw text for the models.
* **`src/eda.py`** (Exploratory Data Analysis): Analyzes the dataset before training. It generates word clouds, sentence length distributions, and class balance charts to help understand the nature of AI vs. Human texts.
* **`src/train.py`**: The core training pipeline. It includes dynamic model selection, gradient scaling (for VRAM optimization), early stopping, and automatic checkpointing to save the best model weights (`.pt`).
* **`src/evaluate.py`**: The testing and metrics suite. Loads the trained model to generate the Confusion Matrix, Precision/Recall/F1-scores, and creates **LIME (Explainable AI)** reports to show exactly *why* the model made a decision.
* **`app.py`**: The interactive web dashboard built with Streamlit. It utilizes batch processing to deliver real-time predictions and LIME visualizations directly in your browser.

---

## 🚀 Installation & Setup

To ensure 100% reproducibility, this project uses a strict dependency pinning strategy.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/nlp-ai-human-detector.git](https://github.com/your-username/nlp-ai-human-detector.git)
cd nlp-ai-human-detector
```
### 2. Set Up the Virtual Environment
It is highly recommended to use a virtual environment to avoid package conflicts.

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate
```
### 3. Install Dependencies
Install all required libraries using the provided requirements file:

** For macOS / Linux:**
```bash
pip install -r requirements.txt
```
** For Windows / Linux:**
```bash
pip install -r requirements_windows.txt
```

### 4. Download GloVe Vectors (Required for HAN)
Download glove.6B.zip from the Official Stanford NLP Group.

Extract glove.6B.100d.txt and place it in the results/saved_models/ directory.

💻 Usage Guide
Make sure your virtual environment is activated before running any commands.

### Step 1: Exploratory Data Analysis (Optional)
To visualize the dataset statistics and generate graphs:

```bash
python -m src.eda
```
### Step 2: Training the Model
The training pipeline is dynamic and supports multiple architectures. The system will automatically monitor validation accuracy, apply early stopping if necessary, and save the best weights to `results/saved_models/best_[model_name].pt`.

**Available Architecture Choices (`--model`):**
`fasttext` | `bilstm` | `charcnn` | `han` | `han_glove` | `roberta` | `distilbert`

```bash
# 1. Train the highly optimized HAN with GloVe (⭐ Recommended)
python -m src.train --model han_glove --epochs 5 --lr 2e-4

# 2. Train a standard RNN architecture
python -m src.train --model bilstm --epochs 5 --lr 2e-4

# 3. Train a Lightweight Transformer architecture
python -m src.train --model distilbert --epochs 3 --lr 2e-5
```

### Step 3: Evaluating the Model
Generate performance metrics, confusion matrices, and Explainable AI (LIME) reports on the test data:

```bash
# Evaluate the GloVe-supported HAN
python -m src.evaluate --model han_glove

# Evaluate the BiLSTM model
python -m src.evaluate --model bilstm

# Evaluate the DistilBERT model
python -m src.evaluate --model distilbert
```
### Step 4: Launching the Web Application
Start the interactive Streamlit dashboard to test the model with your own texts in real-time:

```bash
streamlit run app.py
```
