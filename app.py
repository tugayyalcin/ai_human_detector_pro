import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize 
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer 

# Kendi modüllerimiz
from src.config import DEVICE, DATASET_PATH, MODELS_DIR, MAX_SEQ_LENGTH
from src.dataset import Vocabulary
from src.preprocessing import clean_text_for_model

# Tüm Model Mimarilerini İçe Aktarıyoruz (DeBERTa silindi)
from src.models.fasttext import FastTextClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.han import HANClassifier  # İçe aktarma yolu düzeltildi
from src.models.charcnn import CharCNNClassifier
from src.models.roBERTa import RobertaClassifier
from src.models.distilbert import DistilBertClassifier 

# --- SAYFA AYARLARI VE CSS ---
st.set_page_config(page_title="AI vs Human Detector Pro", page_icon="🕵️‍♂️", layout="wide")

def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stTextArea textarea { border-radius: 10px; border: 1px solid #ddd; }
        .prediction-card {
            padding: 20px;
            border-radius: 15px;
            background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- DİNAMİK VE HIZLI MODEL YÜKLEME ---
@st.cache_resource(show_spinner=False)
def load_system(model_name):
    vocab = None
    # DeBERTa listeden çıkarıldı
    if model_name in ['roberta', 'distilbert']:
        if model_name == 'roberta':
            vocab = AutoTokenizer.from_pretrained('roberta-base')
        elif model_name == 'distilbert':
            vocab = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        is_char = (model_name == "charcnn")
        vocab_name = "vocab_char.pkl" if is_char else "vocab.pkl"
        vocab_path = os.path.join(MODELS_DIR, vocab_name)
        
        if not os.path.exists(vocab_path):
            return "NO_VOCAB", None
            
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    
    if model_name == "fasttext":
        model = FastTextClassifier(len(vocab))
    elif model_name == "bilstm":
        model = BiLSTMClassifier(len(vocab))
    # YENİ: han ve han_glove aynı model sınıfını kullanır, ağırlıklar .pt'den gelir
    elif model_name in ["han", "han_glove"]:
        model = HANClassifier(len(vocab))
    elif model_name == "charcnn":
        model = CharCNNClassifier(len(vocab))
    elif model_name == "roberta":
        model = RobertaClassifier()
    elif model_name == "distilbert": 
        model = DistilBertClassifier()

    model_path = os.path.join(MODELS_DIR, f"best_{model_name}.pt")
    if not os.path.exists(model_path):
        return vocab, None
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return vocab, model

# --- TAHMİN FONKSİYONU (BATCH PROCESSING AKTİF) ---
def predict_proba_for_lime(texts, model, vocab, model_name):
    """LIME'ın ürettiği binlerce metni batch (toplu) halinde GPU'ya gönderir."""
    model.eval()
    all_probs = []
    batch_size = 128  # GPU için optimize edilmiş paket boyutu
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 1. TRANSFORMER MODELLERİ (DeBERTa çıkarıldı)
        if model_name in ['roberta', 'distilbert']:
            encoding = vocab(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
                
        # 2. HAN MODELİ (han ve han_glove entegrasyonu)
        elif model_name in ['han', 'han_glove']:
            max_sentences = 15
            max_words_per_sent = 20
            batch_tensors = np.zeros((len(batch_texts), max_sentences, max_words_per_sent), dtype=np.int64)
            
            for b_idx, text in enumerate(batch_texts):
                # han_glove için dataset'teki aynı temizleme mantığını kullanıyoruz
                dataset_model_type = "han" if model_name == "han_glove" else model_name
                cleaned = clean_text_for_model(text, model_type=dataset_model_type)
                sentences = sent_tokenize(cleaned)[:max_sentences]
                for s_idx, sentence in enumerate(sentences):
                    words = word_tokenize(sentence.lower())[:max_words_per_sent]
                    for w_idx, word in enumerate(words):
                        batch_tensors[b_idx, s_idx, w_idx] = vocab.word2idx.get(word, vocab.word2idx["<UNK>"])
            
            input_tensor = torch.tensor(batch_tensors, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)

        # 3. DİĞER KLASİK MODELLER (1 Boyutlu)
        else:
            batch_tensors = []
            for text in batch_texts:
                cleaned = clean_text_for_model(text, model_type=model_name)
                if model_name == "charcnn":
                    tokens = list(cleaned)
                else:
                    tokens = word_tokenize(cleaned)
                
                token_ids = [vocab.word2idx.get(t, vocab.word2idx["<UNK>"]) for t in tokens][:MAX_SEQ_LENGTH]
                token_ids += [vocab.word2idx["<PAD>"]] * (MAX_SEQ_LENGTH - len(token_ids))
                batch_tensors.append(token_ids)
            
            input_tensor = torch.tensor(batch_tensors, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
                
    return np.array(all_probs)

# --- ARAYÜZ (UI) ---
st.title("🕵️‍♂️ AI vs Human Text Detector")
st.caption("BIM432 NLP Projesi | M4 İşlemci, Toplu İşleme ve LIME Entegrasyonu")

selected_model = st.selectbox(
    "🧠 Test Etmek İstediğiniz Modeli Seçin:",
    ("fasttext", "bilstm", "charcnn", "han", "han_glove", "roberta", "distilbert"), 
    index=4,  # Yeni yıldızımız han_glove varsayılan olarak seçili gelsin
    help="Eğitilmiş modeller arasında geçiş yapabilirsiniz."
)

with st.spinner(f"{selected_model.upper()} modeli belleğe yükleniyor..."):
    vocab, model = load_system(selected_model)

if vocab == "NO_VOCAB":
    st.error(f"🚨 `{selected_model}` için sözlük bulunamadı! Lütfen önce eğitimi başlatın.")
elif model is None:
    st.error(f"🚨 `{selected_model}` modeli için eğitilmiş ağırlık (.pt) bulunamadı!")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Analiz edilecek metni girin:", height=250, 
                                  placeholder="İngilizce metni buraya yapıştırın...")
        
        st.markdown("#### ⚙️ Analiz Ayarları")
        lime_features = st.slider(
            "LIME Hassasiyeti (İncelenecek Kelime Sayısı):", 
            min_value=5, max_value=50, value=15, step=5
        )
        
        analyze_btn = st.button("🚀 Metni Analiz Et", type="primary", use_container_width=True)

    with col2:
        st.info(f"**Aktif Model: {selected_model.upper()}**\nToplu işlem (Batch Processing) aktif.")

    if analyze_btn:
        if len(user_input.strip()) < 15:
            st.warning("⚠️ Lütfen daha uzun bir metin girin.")
        else:
            with st.status(f"🔍 {selected_model.upper()} analiz ediyor...", expanded=True) as status:
                probs = predict_proba_for_lime([user_input], model, vocab, selected_model)[0]
                status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)

            human_prob, ai_prob = probs[0] * 100, probs[1] * 100

            st.divider()
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if ai_prob > human_prob:
                    st.error(f"### 🤖 Tahmin: YAPAY ZEKA\nOlasılık: %{ai_prob:.2f}")
                else:
                    st.success(f"### 👤 Tahmin: İNSAN\nOlasılık: %{human_prob:.2f}")
            with res_col2:
                st.write("Sınıf Olasılıkları")
                st.progress(int(ai_prob), text=f"AI: %{ai_prob:.2f}")
                st.progress(int(human_prob), text=f"İnsan: %{human_prob:.2f}")

            # --- HIZLANDIRILMIŞ LIME ---
            st.divider()
            st.subheader("🧠 Karar Analizi (Explainable AI)")
            with st.spinner("LIME raporu hazırlanıyor (M4 Batch Processing ile roket hızında)..."):
                explainer = LimeTextExplainer(class_names=['Human', 'AI'])
                exp = explainer.explain_instance(
                    user_input, 
                    lambda x: predict_proba_for_lime(x, model, vocab, selected_model), 
                    num_features=lime_features,
                    num_samples=1000 
                )
                exp_html = exp.as_html()
                components.html(exp_html, height=500, scrolling=True)