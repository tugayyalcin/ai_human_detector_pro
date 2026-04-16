import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        İki Yönlü LSTM (BiLSTM) Modeli
        
        Args:
            vocab_size (int): Sözlükteki toplam kelime sayısı
            embed_dim (int): Kelime vektörlerinin (embedding) boyutu
            hidden_dim (int): LSTM hücrelerinin gizli katman boyutu
            num_layers (int): Üst üste dizilecek LSTM katman sayısı
            dropout (float): Aşırı öğrenmeyi (overfitting) engellemek için dropout oranı
        """
        super(BiLSTMClassifier, self).__init__()
        
        # 1. Embedding Katmanı: Kelime ID'lerini yoğun (dense) vektörlere çevirir
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # 2. BiLSTM Katmanı
        # batch_first=True: Giriş tensörünün (Batch Size, Sequence Length, Features) formatında olmasını sağlar.
        # bidirectional=True: Metni hem ileri hem geri yönde okur.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Dropout Katmanı (Regularization için)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Sınıflandırma Katmanı (Fully Connected Layer)
        # BiLSTM kullandığımız için hidden_dim boyutunu 2 ile çarpıyoruz (ileri + geri)
        self.fc = nn.Linear(hidden_dim * 2, 2) # Çıktı boyutu 2: [Yapay Zeka (1), İnsan (0)]

    def forward(self, input_ids):
        """
        İleri besleme (Forward pass) fonksiyonu.
        
        Args:
            input_ids (Tensor): [Batch Size, Sequence Length] boyutlarında kelime ID'leri.
        """
        # Kelimeleri vektörlere dönüştür -> Boyut: [Batch Size, Seq Len, Embed Dim]
        embedded = self.embedding(input_ids)
        
        # LSTM'den geçir
        # lstm_out: Tüm zaman adımlarındaki gizli durumlar
        # (hidden, cell): Son zaman adımındaki gizli ve hücre durumları
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # BiLSTM'in son anındaki ileri ve geri yöndeki çıktılarını birleştiriyoruz.
        # hidden[-2, :, :] -> İleri yöndeki son durum
        # hidden[-1, :, :] -> Geri yöndeki son durum
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Dropout uygula
        dropped_out = self.dropout(hidden_cat)
        
        # Sınıflandırma katmanından geçir ve Logits (ham skorlar) elde et
        logits = self.fc(dropped_out)
        
        return logits

# Modeli test etmek için ufak bir blok (Sadece bu dosya çalıştırıldığında aktif olur)
if __name__ == "__main__":
    # Test amaçlı sahte (dummy) veriler
    dummy_vocab_size = 5000
    dummy_batch = torch.randint(0, dummy_vocab_size, (32, 256)) # Batch Size: 32, Seq Len: 256
    
    model = BiLSTMClassifier(vocab_size=dummy_vocab_size)
    output = model(dummy_batch)
    
    print(f"Model Çıktı Boyutu: {output.shape}") 
    print("Beklenen Boyut: [32, 2] -> (Batch Size, Sınıf Sayısı)")