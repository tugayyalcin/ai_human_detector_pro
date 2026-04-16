import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DebertaClassifier(nn.Module):
    """
    DeBERTa-V3 (Decoding-enhanced BERT with disentangled attention) Modeli.
    Kelimelerin sadece anlamına değil, cümle içindeki göreceli konumlarına da (relative position)
    odaklanarak yapay zekanın yapısal hatalarını çok iyi yakalar.
    """
    def __init__(self, model_name='microsoft/deberta-v3-base', dropout=0.3):
        super(DebertaClassifier, self).__init__()
        
        # DeBERTa modelini otomatik yükle
        self.deberta = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Dropout ve Sınıflandırma Katmanı
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config.hidden_size, 2) # [Yapay Zeka (1), İnsan (0)]

    def forward(self, input_ids, attention_mask):
        """İleri besleme (Forward pass)"""
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # DeBERTa'da tüm cümleyi temsil eden [CLS] token'ını (0. indeks) alıyoruz.
        # Boyut: [Batch Size, Hidden Size]
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Sınıflandırma
        x = self.dropout(cls_output)
        logits = self.fc(x)
        
        return logits

# Test Bloğu
if __name__ == "__main__":
    dummy_input_ids = torch.randint(0, 128000, (8, 256)) # DeBERTa v3 vocab size: ~128000
    dummy_attention_mask = torch.ones((8, 256))
    
    print("DeBERTa-v3 modeli yükleniyor... (Bu model biraz ağırdır, indirmesi sürebilir)")
    model = DebertaClassifier()
    output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print(f"DeBERTa Çıktı Boyutu: {output.shape} -> Beklenen: [8, 2]")