import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer

# Kendi modüllerimiz
from src.config import DEVICE, DATASET_PATH, MODELS_DIR, FIGURES_DIR, BATCH_SIZE
from src.dataset import AIHumanDataset, Vocabulary
from src.models.bilstm import BiLSTMClassifier
from src.models.han import HANClassifier
from src.models.fasttext import FastTextClassifier
from src.models.charcnn import CharCNNClassifier
from src.models.roBERTa import RobertaClassifier
from src.models.deBERTa import DebertaClassifier
from src.models.distilbert import DistilBertClassifier

def evaluate_model():
    parser = argparse.ArgumentParser(description="NLP Model Evaluation")
    # ---> DÜZELTME 1: han_glove listeye eklendi <---
    parser.add_argument("--model", type=str, required=True, choices=["bilstm", "han", "han_glove", "fasttext", "charcnn", "roberta", "deberta", "distilbert"])
    args = parser.parse_args()

    print(f"🔍 {args.model.upper()} Modeli Test Ediliyor...")
    print(f"💻 Cihaz: {DEVICE}")
    
    # 1. Veriyi Yükle
    df = pd.read_csv(DATASET_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_texts = train_df['text'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['generated'].tolist()

    # 2. Vocabulary Hazırlığı
    vocab = None
    if args.model not in ['roberta', 'deberta', 'distilbert']:
        is_char = (args.model == "charcnn")
        vocab = Vocabulary(is_char_level=is_char)
        vocab.build_vocab(train_texts)

    # ---> DÜZELTME 2: Dataset için 'han_glove' ismini 'han' gibi işle <---
    dataset_model_type = "han" if args.model == "han_glove" else args.model

    # 3. Test Dataset
    test_dataset = AIHumanDataset(test_texts, test_labels, dataset_model_type, vocab=vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Modeli Başlat ve Yükle
    model_path = os.path.join(MODELS_DIR, f"best_{args.model}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Hata: {model_path} bulunamadı. Lütfen önce modeli eğitin!")

    if args.model == "bilstm":
        model = BiLSTMClassifier(len(vocab))
    elif args.model == "han":
        model = HANClassifier(len(vocab))
    # ---> DÜZELTME 3: han_glove için GloVe txt yüklemeye gerek yok, ağırlıklar .pt'den gelecek <---
    elif args.model == "han_glove":
        model = HANClassifier(len(vocab))
    elif args.model == "fasttext":
        model = FastTextClassifier(len(vocab))
    elif args.model == "charcnn":
        model = CharCNNClassifier(len(vocab))
    elif args.model == "roberta":
        model = RobertaClassifier()
    elif args.model == "deberta":
        model = DebertaClassifier()
    elif args.model == "distilbert":
        model = DistilBertClassifier()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 5. Tahmin ve ERROR ANALYSIS (Hata Analizi) Kayıtları
    all_preds = []
    all_labels = []
    error_analysis_data = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Test Ediliyor")):
            if args.model in ['roberta', 'deberta', 'distilbert']:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
            else:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(input_ids)

            probs = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- ERROR ANALYSIS VERİ TOPLAMA ---
            for j in range(len(labels)):
                true_lbl = labels[j].item()
                pred_lbl = predicted[j].item()
                if true_lbl != pred_lbl:
                    idx = i * BATCH_SIZE + j
                    if idx < len(test_texts):
                        error_analysis_data.append({
                            'Metin': test_texts[idx][:500] + "...", 
                            'Gercek_Etiket': 'AI' if true_lbl == 1 else 'Human',
                            'Tahmin_Edilen': 'AI' if pred_lbl == 1 else 'Human',
                            'Eminlik_Orani': f"%{confidences[j].item() * 100:.2f}"
                        })

    # Error Analysis'i CSV Olarak Kaydet
    error_df = pd.DataFrame(error_analysis_data)
    error_csv_path = os.path.join(FIGURES_DIR, f"{args.model}_error_analysis.csv")
    error_df.to_csv(error_csv_path, index=False)
    print(f"\n🔎 Hata Analizi (Error Analysis) kaydedildi: {error_csv_path}")

    # ==========================================
    # 6. METRICS SUMMARY (METRİK ÖZETİ) VE KAYIT
    # ==========================================
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n" + "="*50)
    print(f"📊 {args.model.upper()} - METRICS SUMMARY")
    print("="*50)
    print(f"Doğruluk (Accuracy)  : {acc:.4f}")
    print(f"Hassasiyet (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall)  : {recall:.4f}")
    print(f"F1-Skoru (F1-Score)  : {f1:.4f}")
    print("="*50)

    # ==========================================
    # 6.5 PERFORMANS METRİKLERİ BAR GRAFİĞİ
    # ==========================================
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [acc, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics_names, y=metrics_values, palette="viridis")
    plt.title(f'{args.model.upper()} - Model Performans Metrikleri', fontsize=16)
    plt.ylabel('Skor', fontsize=14)
    plt.ylim(0, 1.15) 

    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=14)

    bar_fig_path = os.path.join(FIGURES_DIR, f"{args.model}_metrics_bar.png")
    plt.savefig(bar_fig_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Performans Bar Grafiği kaydedildi: {bar_fig_path}")

    # 7. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (0)', 'AI (1)'], yticklabels=['Human (0)', 'AI (1)'])
    plt.title(f'{args.model.upper()} - Confusion Matrix')
    cm_path = os.path.join(FIGURES_DIR, f"{args.model}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"📈 Confusion Matrix grafiği kaydedildi: {cm_path}")

    # ==========================================
    # 8. EXPLAINABLE AI (LIME) & FEATURE IMPORTANCE
    # ==========================================
    print("\n🧠 LIME (Explainable AI) ile özellik çıkarımı yapılıyor...")
    
    def predict_proba_wrapper(texts):
        model.eval()
        all_batch_probs = []
        # ---> DÜZELTME 4: LIME dataset için de dataset_model_type kullanılıyor <---
        temp_dataset = AIHumanDataset(texts, [0]*len(texts), dataset_model_type, vocab=vocab)
        temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in temp_loader:
                if args.model in ['roberta', 'deberta', 'distilbert']:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    outputs = model(input_ids, attention_mask)
                else:
                    input_ids = batch['input_ids'].to(DEVICE)
                    outputs = model(input_ids)
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_batch_probs.extend(probs)
        return np.array(all_batch_probs)

    sample_idx = next(idx for idx, label in enumerate(test_labels) if label == 1)
    sample_text = test_texts[sample_idx]

    explainer = LimeTextExplainer(class_names=['Human', 'AI'])
    exp = explainer.explain_instance(sample_text, predict_proba_wrapper, num_features=10)
    
    fig = exp.as_pyplot_figure()
    plt.title(f'{args.model.upper()} - LIME Feature Importance')
    lime_fig_path = os.path.join(FIGURES_DIR, f"{args.model}_lime_explanation.png")
    fig.savefig(lime_fig_path, bbox_inches='tight')
    plt.close(fig)
    
    lime_html_path = os.path.join(FIGURES_DIR, f"{args.model}_lime_report.html")
    exp.save_to_file(lime_html_path)
    
    lime_text_path = os.path.join(FIGURES_DIR, f"{args.model}_lime_sample_text.txt")
    with open(lime_text_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print(f"✨ Explainable AI Grafiği kaydedildi: {lime_fig_path}")

if __name__ == "__main__":
    evaluate_model()