import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np # ---> YENİ: GloVe Matrisi için eklendi
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import pickle

from src.config import DEVICE, DATASET_PATH, MODELS_DIR, FIGURES_DIR, BATCH_SIZE, MAX_SEQ_LENGTH
from src.dataset import AIHumanDataset, Vocabulary
from src.models.bilstm import BiLSTMClassifier
from src.models.han import HANClassifier
from src.models.fasttext import FastTextClassifier
from src.models.charcnn import CharCNNClassifier
from src.models.roBERTa import RobertaClassifier
from src.models.deBERTa import DebertaClassifier
from src.models.distilbert import DistilBertClassifier

# ---> YENİ: GloVe Matrisi Oluşturucu Fonksiyon <---
def create_embedding_matrix(vocab, glove_path, embed_dim=100):
    print("🔍 GloVe vektörleri yükleniyor... (Bu işlem birkaç saniye sürebilir)")
    
    # Kelimeler GloVe'da yoksa rastgele başlasın diye matrisi hazırlıyoruz
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    word2idx = vocab.word2idx
    
    hits = 0
    misses = 0

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word2idx:
                    idx = word2idx[word]
                    embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
                    hits += 1
                else:
                    misses += 1
    except FileNotFoundError:
        print(f"🚨 HATA: GloVe dosyası ({glove_path}) bulunamadı! Model rastgele vektörlerle eğitilecek.")
        return None

    print(f"✅ Başarılı! {hits} kelime GloVe ile eşleşti. {len(vocab) - hits} kelime bulunamadı (rastgele bırakıldı).")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def train_model():
    parser = argparse.ArgumentParser(description="NLP AI-Human Detector Training")
    # ---> YENİ: han_glove seçeneği eklendi <---
    parser.add_argument("--model", type=str, required=True, choices=["bilstm", "han", "han_glove", "fasttext", "charcnn", "roberta", "deberta", "distilbert"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient Accumulation Steps")
    args = parser.parse_args()

    print(f"🛠️  Model Hazırlanıyor: {args.model.upper()}")
    
    use_autocast = True
    if args.model == 'deberta':
        use_autocast = False
        print(f"💻 Cihaz: {DEVICE} (ÖZEL DURUM: DeBERTa NaN Önlemi - Saf FP32 Aktif!)")
    else:
        print(f"💻 Cihaz: {DEVICE} (Otomatik Yarı Hassasiyet & Gradyan Birikimi: AKTİF)")

    df = pd.read_csv(DATASET_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_texts, train_labels = train_df['text'].tolist(), train_df['generated'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['generated'].tolist()

    vocab = None
    if args.model not in ['roberta', 'deberta', 'distilbert']:
        is_char = (args.model == "charcnn")
        vocab = Vocabulary(is_char_level=is_char)
        vocab.build_vocab(train_texts)

        vocab_name = "vocab_char.pkl" if is_char else "vocab.pkl"
        vocab_path = os.path.join(MODELS_DIR, vocab_name)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    # Dataset oluştururken "han_glove" ismini "han" gibi işlemesi için ufak düzeltme
    dataset_model_type = "han" if args.model == "han_glove" else args.model
    train_dataset = AIHumanDataset(train_texts, train_labels, dataset_model_type, vocab=vocab)
    val_dataset = AIHumanDataset(val_texts, val_labels, dataset_model_type, vocab=vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    if args.model == "bilstm":
        model = BiLSTMClassifier(len(vocab)).to(DEVICE)
    elif args.model == "han":
        model = HANClassifier(len(vocab)).to(DEVICE)
    # ---> YENİ: GloVe Destekli HAN Başlatma <---
    elif args.model == "han_glove":
        glove_path = os.path.join(MODELS_DIR, "glove.6B.100d.txt")
        glove_matrix = create_embedding_matrix(vocab, glove_path)
        model = HANClassifier(len(vocab), pretrained_embeddings=glove_matrix).to(DEVICE)
    elif args.model == "fasttext":
        model = FastTextClassifier(len(vocab)).to(DEVICE)
    elif args.model == "charcnn":
        model = CharCNNClassifier(len(vocab)).to(DEVICE)
    elif args.model == "roberta":
        model = RobertaClassifier().to(DEVICE)
    elif args.model == "deberta":
        model = DebertaClassifier().to(DEVICE).float()
    elif args.model == "distilbert":
        model = DistilBertClassifier().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=use_autocast)
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    try: 
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            optimizer.zero_grad() 

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
            for i, batch in enumerate(loop):
                with torch.amp.autocast('cuda', enabled=use_autocast):
                    if args.model in ['roberta', 'deberta', 'distilbert']:
                        input_ids = batch['input_ids'].to(DEVICE)
                        attention_mask = batch['attention_mask'].to(DEVICE)
                        labels = batch['label'].to(DEVICE)
                        outputs = model(input_ids, attention_mask)
                    else:
                        input_ids = batch['input_ids'].to(DEVICE)
                        labels = batch['label'].to(DEVICE)
                        outputs = model(input_ids)

                    loss = criterion(outputs, labels)
                    loss = loss / args.grad_accum

                if not use_autocast:
                    loss.backward()
                    if ((i + 1) % args.grad_accum == 0) or ((i + 1) == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    scaler.scale(loss).backward()
                    if ((i + 1) % args.grad_accum == 0) or ((i + 1) == len(train_loader)):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                total_loss += (loss.item() * args.grad_accum)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(loss=(loss.item() * args.grad_accum), acc=100.*correct/total)

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    with torch.amp.autocast('cuda', enabled=use_autocast):
                        if args.model in ['roberta', 'deberta', 'distilbert']:
                            input_ids = batch['input_ids'].to(DEVICE)
                            attention_mask = batch['attention_mask'].to(DEVICE)
                            labels = batch['label'].to(DEVICE)
                            outputs = model(input_ids, attention_mask)
                        else:
                            input_ids = batch['input_ids'].to(DEVICE)
                            labels = batch['label'].to(DEVICE)
                            outputs = model(input_ids)

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(total_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"📊 Epoch {epoch+1} Bitti - Val Acc: %{val_acc:.2f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # İsimlendirme tutarlılığı için han_glove modelini han_glove olarak kaydedelim
                save_name = "best_han_glove.pt" if args.model == "han_glove" else f"best_{args.model}.pt"
                save_path = os.path.join(MODELS_DIR, save_name)
                torch.save(model.state_dict(), save_path)
                print(f"💾 Yeni en iyi model kaydedildi: {save_path}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n🛑 Eğitim kullanıcı tarafından (Ctrl+C) manuel olarak durduruldu (Early Stopping)!")
        print("Mevcut verilerle grafikler çiziliyor, lütfen bekleyin...\n")

    actual_epochs = len(history['train_loss'])
    if actual_epochs > 0:
        epochs_range = range(1, actual_epochs + 1)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', color='red')
        plt.title(f'{args.model.upper()} - Training Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(epochs_range)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o', color='blue')
        plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o', color='green')
        plt.title(f'{args.model.upper()} - Accuracy', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(epochs_range)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        history_fig_path = os.path.join(FIGURES_DIR, f"{args.model}_training_history.png")
        plt.savefig(history_fig_path, bbox_inches='tight')
        plt.close()
        print(f"📈 Eğitim eğrileri grafiği kaydedildi: {history_fig_path}")
        
    print("🚀 Eğitim Tamamlandı!")

if __name__ == "__main__":
    train_model()