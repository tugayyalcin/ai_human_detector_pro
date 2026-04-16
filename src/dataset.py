import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# Özel modüllerimiz
from .preprocessing import clean_text_for_model
from .config import MAX_SEQ_LENGTH

class Vocabulary:
    """BiLSTM, FastText, HAN ve Char-CNN için kelime/harf sözlüğü oluşturur."""
    def __init__(self, is_char_level=False):
        self.is_char_level = is_char_level
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.idx = 2

        # Char-CNN için alfabeyi ve noktalama işaretlerini baştan ekleyebiliriz
        if is_char_level:
            chars = string.ascii_lowercase + string.digits + string.punctuation + " "
            for char in chars:
                self.add_token(char)

    def add_token(self, token):
        if token not in self.word2idx:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def build_vocab(self, texts):
        for text in texts:
            if self.is_char_level:
                tokens = list(text.lower())
            else:
                tokens = word_tokenize(text.lower())
            
            for token in tokens:
                self.add_token(token)

    def __len__(self):
        return len(self.word2idx)


class AIHumanDataset(Dataset):
    """6 farklı modele uygun veri sağlayan ana PyTorch Dataset sınıfımız."""
    def __init__(self, texts, labels, model_type, vocab=None, max_seq_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        
        # ---> YENİ: Tüm Transformer modelleri için dinamik Tokenizer (distilbert eklendi) <---
        if self.model_type in ['bert', 'roberta', 'deberta', 'distilbert']:
            if self.model_type == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            elif self.model_type == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            elif self.model_type == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
            elif self.model_type == 'distilbert':
                self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        # HAN (Hiyerarşik model) için özel limitler
        self.max_sentences = 15
        self.max_words_per_sent = 20

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 1. Adım: Metni modele özel temizle
        cleaned_text = clean_text_for_model(raw_text, self.model_type)

        # 2. Adım: Modele göre tensör (matris) oluştur
        # ---> YENİ: distilbert de buraya girecek <---
        if self.model_type in ['bert', 'roberta', 'deberta', 'distilbert']:
            encoding = self.tokenizer(
                cleaned_text,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': label
            }

        elif self.model_type == 'han':
            # HAN için 2 boyutlu matris: [Cümle Sayısı x Cümledeki Kelime Sayısı]
            sentences = sent_tokenize(cleaned_text)[:self.max_sentences]
            doc_tensor = np.zeros((self.max_sentences, self.max_words_per_sent), dtype=np.int64)
            
            for i, sentence in enumerate(sentences):
                words = word_tokenize(sentence.lower())[:self.max_words_per_sent]
                for j, word in enumerate(words):
                    doc_tensor[i, j] = self.vocab.word2idx.get(word, self.vocab.word2idx["<UNK>"])
            
            return {'input_ids': torch.tensor(doc_tensor, dtype=torch.long), 'label': label}

        else:
            # BiLSTM, FastText (Kelime bazlı) ve Char-CNN (Harf bazlı)
            if self.model_type == 'charcnn':
                tokens = list(cleaned_text.lower())
            else:
                tokens = word_tokenize(cleaned_text)

            # Tokenleri indekslere çevir (Maksimum uzunluğa göre kırp)
            token_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx["<UNK>"]) for t in tokens][:self.max_seq_len]
            
            # Padding (Boşlukları sıfır ile doldur)
            padding_length = self.max_seq_len - len(token_ids)
            token_ids.extend([self.vocab.word2idx["<PAD>"]] * padding_length)

            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'label': label
            }