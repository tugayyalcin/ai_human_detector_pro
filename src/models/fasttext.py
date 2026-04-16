import torch
import torch.nn as nn
import torch.nn.functional as F

class FastTextClassifier(nn.Module):
    """
    Derin Öğrenme Tabanlı FastText Modeli
    Kelime vektörlerinin ortalamasını (Global Average Pooling) alarak hızlı ve etkili sınıflandırma yapar.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3):
        """
        Args:
            vocab_size (int): Sözlükteki toplam kelime sayısı
            embed_dim (int): Kelime vektörlerinin boyutu
            hidden_dim (int): Gizli katman boyutu
            dropout (float): Aşırı öğrenmeyi engellemek için oran
        """
        super(FastTextClassifier, self).__init__()
        
        # Kelime Embedding (Vektör) Katmanı (Padding için 0. index sıfır vektörü üretir)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # Sınıflandırma Katmanları
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2) # [Yapay Zeka (1), İnsan (0)]

    def forward(self, input_ids):
        """
        İleri besleme fonksiyonu.
        input_ids boyutu: [Batch Size, Sequence Length]
        """
        # 1. Kelimeleri vektörlere çevir -> Boyut: [Batch Size, Seq Len, Embed Dim]
        embedded = self.embedding(input_ids)
        
        # --- HARİKA MÜHENDİSLİK DETAYI: Padding Maskeleme ---
        # Kısa cümleleri doldurmak için attığımız 0'ların (PAD) ortalamayı bozmasını engelliyoruz.
        
        # Gerçek kelimelerin olduğu yerlere 1, PAD olan yerlere 0 koyan bir maske oluştur
        # Boyut: [Batch Size, Seq Len, 1]
        mask = (input_ids != 0).float().unsqueeze(-1)
        
        # Embedding'leri maske ile çarp (PAD'ler kesin olarak 0 olur)
        embedded = embedded * mask
        
        # Cümledeki gerçek kelimelerin vektörlerini topla
        summed = embedded.sum(dim=1)
        
        # Her cümlede kaç tane "gerçek" kelime olduğunu say (0'a bölünme hatasını engellemek için clamp kullanıyoruz)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        
        # Sadece gerçek kelime sayısına bölerek KUSURSUZ ORTALAMAYI bul (Global Average Pooling)
        # Boyut: [Batch Size, Embed Dim]
        pooled = summed / counts
        # ----------------------------------------------------
        
        # 2. Sınıflandırma Ağı
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# Modeli test etmek için ufak bir blok
if __name__ == "__main__":
    dummy_vocab_size = 5000
    dummy_batch = torch.randint(0, dummy_vocab_size, (32, 256)) # Batch Size: 32, Seq Len: 256
    
    # Bilerek bazı yerleri PAD (0) yapalım ki maskeleme test edilsin
    dummy_batch[:, 100:] = 0 
    
    model = FastTextClassifier(vocab_size=dummy_vocab_size)
    output = model(dummy_batch)
    
    print(f"FastText Model Çıktı Boyutu: {output.shape}") 
    print("Beklenen Boyut: [32, 2] -> (Batch Size, Sınıf Sayısı)")