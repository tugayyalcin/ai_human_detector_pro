import torch
import os

# --- CİHAZ (DEVICE) AYARLARI ---
# Apple Silicon (M serisi) için MPS (Metal Performance Shaders) kontrolü
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Apple Silicon (MPS) aktif! M4 çipinin gücü kullanılıyor.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 NVIDIA GPU (CUDA) aktif!")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Uyarı: Sadece CPU kullanılıyor.")

# --- DİZİN YOLLARI ---
# Mevcut dosyanın (config.py) bulunduğu dizinin bir üstü ana proje dizinidir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Veri setini koyacağımız yer
DATASET_PATH = os.path.join(DATA_DIR, "ai_human_dataset.csv")

# Kod çalıştığında gerekli klasörlerin var olduğundan emin ol (yoksa otomatik oluştur)
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- GENEL EĞİTİM HİPERPARAMETRELERİ ---
# Robert ve Dilbert için 8 , diğerleri için 32
BATCH_SIZE = 8          # Uzunluğu kısalttığımız için artık 8'e yer açıldı! 
MAX_EPOCHS = 10
LEARNING_RATE = 2e-5  # BERT için ideal, diğer modeller için train.py'da değiştireceğiz
MAX_SEQ_LENGTH = 256  # BERT, BiLSTM ve HAN için metin kırpma/padding uzunluğu

# --- MODELLER ---
MODEL_NAMES = ["fasttext", "charcnn", "bilstm", "han", "bert"]