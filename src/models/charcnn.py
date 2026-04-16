import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNNClassifier(nn.Module):
    """
    Karakter Seviyesi Evrişimli Sinir Ağı (Char-CNN)
    Metinleri harf harf okuyarak n-gram karakter örüntülerini yakalar.
    """
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, kernel_sizes=[3, 4, 5], dropout=0.5):
        """
        Args:
            vocab_size (int): Sözlükteki toplam karakter/harf sayısı (Alfabedeki harfler + noktalama)
            embed_dim (int): Karakterlerin vektör boyutu (Kelimelere göre daha küçük seçilir)
            num_filters (int): Her bir kernel boyutu için üretilecek özellik haritası (filtre) sayısı
            kernel_sizes (list): Yan yana okunacak harf grupları (Örn: 3'lü, 4'lü, 5'li harf öbekleri)
            dropout (float): Ezberlemeyi önleme oranı (CNN'lerde genelde 0.5 kullanılır)
        """
        super(CharCNNClassifier, self).__init__()
        
        # 1. Karakter Embedding Katmanı (Padding için 0. index sıfır vektörü)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # 2. Paralel 1D Konvolüsyon (Evrişim) Katmanları
        # nn.ModuleList, Pytorch'a bu katmanların bir liste halinde modelleneceğini söyler
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        # 3. Sınıflandırma Katmanları
        self.dropout = nn.Dropout(dropout)
        # Toplam filtre sayısı = filtre sayısı (128) * farklı kernel sayısı (3) = 384
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 2) # [Yapay Zeka (1), İnsan (0)]

    def forward(self, input_ids):
        """
        İleri besleme fonksiyonu.
        input_ids boyutu: [Batch Size, Sequence Length]
        """
        # 1. Karakterleri vektörlere çevir
        # Boyut: [Batch Size, Sequence Length, Embed Dim]
        embedded = self.embedding(input_ids)
        
        # PyTorch'ta Conv1d katmanı, özellikleri (kanalları) 2. boyutta bekler.
        # Bu yüzden Sequence Length ile Embed Dim'in yerini değiştiriyoruz.
        # Yeni Boyut: [Batch Size, Embed Dim, Sequence Length]
        embedded = embedded.permute(0, 2, 1)
        
        # 2. Konvolüsyon ve Havuzlama (Pooling) İşlemleri
        conv_results = []
        for conv in self.convs:
            # Evrişim uygula ve ReLU aktivasyonundan geçir
            x = F.relu(conv(embedded))
            
            # Global Max Pooling (1D): Her bir filtrenin yakaladığı en güçlü özelliği (max değeri) al
            # Boyut: [Batch Size, Num Filters, Feature Map Length] -> [Batch Size, Num Filters]
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            
            conv_results.append(x)
        
        # 3. Tüm farklı filtrelerden (3'lü, 4'lü, 5'li harf) gelen özellikleri birleştir (Concatenate)
        # Boyut: [Batch Size, Num Filters * len(kernel_sizes)]
        cat = torch.cat(conv_results, dim=1)
        
        # 4. Sınıflandırma
        cat = self.dropout(cat)
        logits = self.fc(cat)
        
        return logits

# Modeli test etmek için ufak bir blok
if __name__ == "__main__":
    # Char-CNN için vocab_size genelde küçüktür (Örn: 100 civarı karakter)
    dummy_vocab_size = 100 
    dummy_batch = torch.randint(0, dummy_vocab_size, (32, 512)) # Batch: 32, Max Harf Sayısı: 512
    
    model = CharCNNClassifier(vocab_size=dummy_vocab_size)
    output = model(dummy_batch)
    
    print(f"Char-CNN Model Çıktı Boyutu: {output.shape}") 
    print("Beklenen Boyut: [32, 2] -> (Batch Size, Sınıf Sayısı)")