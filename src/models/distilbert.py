import torch
import torch.nn as nn
from transformers import AutoModel

class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DistilBertClassifier, self).__init__()
        # DistilBERT, RoBERTa gibi çalışır
        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Metnin anlamını taşıyan ilk tokeni (CLS) alıyoruz
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.fc(output)