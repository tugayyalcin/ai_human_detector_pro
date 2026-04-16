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

```bash
pip install -r requirements.txt
```
(Note: If your file is named requirements_windows.txt, use that name in the command instead).

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
Train the GloVe-supported Hierarchical Attention Network. The system will automatically save the best model to results/saved_models/best_han_glove.pt:

```bash
python -m src.train --model han_glove --epochs 5 --lr 2e-4
```
### Step 3: Evaluating the Model
Generate performance metrics, confusion matrices, and Explainable AI (LIME) reports on the test data:

```bash
python -m src.evaluate --model han_glove
```
### Step 4: Launching the Web Application
Start the interactive Streamlit dashboard to test the model with your own texts in real-time:

```bash
streamlit run app.py
```