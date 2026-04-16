import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RobertaClassifier(nn.Module):
    """
    RoBERTa (Robustly Optimized BERT Pretraining Approach) Sınıflandırma Modeli.
    Klasik BERT'e göre daha büyük bir veri setiyle ve optimize edilmiş hiperparametrelerle eğitilmiştir.
    """
    def __init__(self, model_name='roberta-base', dropout=0.3):
        super(RobertaClassifier, self).__init__()
        
        # RoBERTa modelini ve konfigürasyonunu otomatik yükle
        self.roberta = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Dropout ve Sınıflandırma Katmanı
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config.hidden_size, 2) # [Yapay Zeka (1), İnsan (0)]

    def forward(self, input_ids, attention_mask):
        """İleri besleme (Forward pass)"""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # RoBERTa'da tüm cümleyi temsil eden özel token <s> (0. indeks) kullanılır.
        # Boyut: [Batch Size, Hidden Size]
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Sınıflandırma
        x = self.dropout(cls_output)
        logits = self.fc(x)
        
        return logits

# Test Bloğu
if __name__ == "__main__":
    dummy_input_ids = torch.randint(0, 50265, (8, 256)) # RoBERTa vocab size: ~50265
    dummy_attention_mask = torch.ones((8, 256))
    
    print("RoBERTa modeli yükleniyor...")
    model = RobertaClassifier()
    output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print(f"RoBERTa Çıktı Boyutu: {output.shape} -> Beklenen: [8, 2]")