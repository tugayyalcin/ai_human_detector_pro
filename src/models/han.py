import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Hem kelime hem de cümle seviyesinde kullanılacak ortak Dikkat (Attention) Mekanizması"""
    def __init__(self, hidden_dim, context_dim):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, context_dim)
        # Rastgele başlatılan bağlam vektörü (neyin "önemli" olduğunu öğrenir)
        self.context_vector = nn.Parameter(torch.rand(context_dim))

    def forward(self, x):
        # x boyutu: [Batch Size, Sequence Length, Hidden Dim]
        
        # 1. Temsilleri MLP'den geçirip tanh aktivasyonu uygula
        u = torch.tanh(self.linear(x)) 
        
        # 2. Bağlam vektörü ile çarpıp Softmax ile ağırlıkları bul
        # Ağırlıklar, kelimenin veya cümlenin ne kadar önemli olduğunu gösterir
        alpha = F.softmax(torch.matmul(u, self.context_vector), dim=1) 
        
        # 3. Ağırlıklarla orijinal çıktıları çarp (Ağırlıklı Toplam)
        attended_output = torch.bmm(alpha.unsqueeze(1), x).squeeze(1) 
        
        return attended_output, alpha

class WordLevelRNN(nn.Module):
    """Kelimelerden Cümle Temsilleri Çıkaran Ağ"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_dim, pretrained_embeddings=None):
        super(WordLevelRNN, self).__init__()
        
        # ---> YENİ: GloVe Vektörlerini Yükleme Mantığı <---
        if pretrained_embeddings is not None:
            # Dışarıdan gelen matrisi kullan. freeze=False yaparak modelin 
            # bu vektörleri kendi görevimize (AI Detector) göre ince ayar yapmasına izin veriyoruz.
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            print("🚀 HAN: GloVe Vektörleri Başarıyla Yüklendi!")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = AttentionLayer(hidden_dim * 2, context_dim)

    def forward(self, word_inputs):
        embedded = self.embedding(word_inputs)
        gru_out, _ = self.gru(embedded)
        sentence_vector, _ = self.attention(gru_out)
        return sentence_vector

class HANClassifier(nn.Module):
    """Hiyerarşik Dikkat Ağı (HAN) Ana Modeli"""
    def __init__(self, vocab_size, embed_dim=100, word_hidden_dim=50, sent_hidden_dim=50, context_dim=100, dropout=0.3, pretrained_embeddings=None):
        super(HANClassifier, self).__init__()
        
        # 1. Aşama: Kelime Seviyesi (GloVe vektörlerini buraya paslıyoruz)
        self.word_rnn = WordLevelRNN(vocab_size, embed_dim, word_hidden_dim, context_dim, pretrained_embeddings)
        
        # 2. Aşama: Cümle Seviyesi
        self.sent_gru = nn.GRU(word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attention = AttentionLayer(sent_hidden_dim * 2, context_dim)
        
        # 3. Aşama: Sınıflandırma
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sent_hidden_dim * 2, 2) 

    def forward(self, input_ids):
        batch_size, max_sentences, max_words = input_ids.shape
        
        # Kelime RNN'ine verebilmek için matrisi düzleştir
        word_inputs = input_ids.view(batch_size * max_sentences, max_words)
        
        # Her bir cümle için tek bir vektör elde et
        sentence_vectors = self.word_rnn(word_inputs) 
        
        # Cümle vektörlerini tekrar asıl boyutlarına ayır
        sentence_vectors = sentence_vectors.view(batch_size, max_sentences, -1)
        
        # Cümleleri GRU'dan geçir ve dikkat mekanizmasını uygula
        sent_gru_out, _ = self.sent_gru(sentence_vectors)
        doc_vector, _ = self.sent_attention(sent_gru_out)
        
        # Sonuç vektörüne Dropout uygula ve sınıflandır
        doc_vector = self.dropout(doc_vector)
        logits = self.fc(doc_vector)
        
        return logits

# Test Bloğu
if __name__ == "__main__":
    # Test amaçlı sahte veriler
    dummy_vocab_size = 5000
    dummy_batch = torch.randint(0, dummy_vocab_size, (32, 15, 20)) 
    
    # 1. Normal Başlatma Testi
    model_normal = HANClassifier(vocab_size=dummy_vocab_size)
    print("Normal HAN Çıktı Boyutu:", model_normal(dummy_batch).shape)

    # 2. Sahte GloVe Matrisi ile Test
    dummy_glove = torch.randn(dummy_vocab_size, 100)
    model_glove = HANClassifier(vocab_size=dummy_vocab_size, pretrained_embeddings=dummy_glove)
    print("GloVe'lu HAN Çıktı Boyutu:", model_glove(dummy_batch).shape)