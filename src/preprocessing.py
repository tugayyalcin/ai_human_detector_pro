import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK paketlerini ilk çalıştırmada otomatik indir
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        print("NLTK verileri indiriliyor...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()
STOPWORDS = set(stopwords.words('english'))

def clean_text_for_model(text: str, model_type: str) -> str:
    """
    Kullanılacak modele göre metin temizleme işlemini özelleştirir.
    
    Args:
        text (str): Temizlenecek ham metin.
        # ---> YENİ: distilbert listeye eklendi <---
        model_type (str): 'fasttext', 'bilstm', 'han', 'charcnn', 'bert', 'roberta', 'deberta', 'distilbert'
    """
    if not isinstance(text, str):
        return ""

    # Tüm modeller için ortak temizlik (HTML etiketleri ve Linkler)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Fazladan boşlukları tek boşluğa indirge
    text = re.sub(r'\s+', ' ', text).strip()

    # Model spesifik temizlik adımları
    # ---> YENİ: distilbert VIP listesine eklendi <---
    if model_type in ['bert', 'roberta', 'deberta', 'distilbert', 'charcnn']:
        # Transformer ve Char-CNN modelleri orijinal bağlama ve noktalama işaretlerine ihtiyaç duyar.
        # Sadece temel temizlik yapıp bırakıyoruz.
        return text

    elif model_type in ['fasttext', 'bilstm', 'han']:
        # Bu modeller için küçük harf, noktalama ve stopword temizliği faydalıdır.
        text = text.lower()
        # Noktalama işaretlerini kaldır
        text = re.sub(r'[^\w\s]', '', text)
        
        # Stopwords (Dolgu kelimeleri) temizle
        words = word_tokenize(text)
        words = [word for word in words if word not in STOPWORDS]
        text = " ".join(words)
        
        return text

    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")

# Test için küçük bir blok (Sadece bu dosya doğrudan çalıştırıldığında çalışır)
if __name__ == "__main__":
    sample_ai_text = "Hello there! <br> I am an AI, and I'm 100% sure this is a test: https://test.com. Are you ready?"
    
    print("Orijinal Metin:", sample_ai_text)
    print("-" * 50)
    print("DistilBERT/RoBERTa İçin:", clean_text_for_model(sample_ai_text, "distilbert"))
    print("BiLSTM/FastText İçin:", clean_text_for_model(sample_ai_text, "bilstm"))