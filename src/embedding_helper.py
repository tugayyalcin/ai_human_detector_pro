import numpy as np
import pickle
import os

def create_embedding_matrix(vocab_path, glove_path, embed_dim=100):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    word2idx = vocab.word2idx
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    
    hits = 0
    misses = 0

    print("🔍 GloVe vektörleri yükleniyor...")
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
                
    print(f"✅ Bitti! {hits} kelime GloVe ile eşleşti. {len(vocab) - hits} kelime rastgele bırakıldı.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)