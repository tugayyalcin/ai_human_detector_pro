import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Kendi modüllerimiz
from src.config import DATASET_PATH, FIGURES_DIR

def run_eda():
    print("🔍 Veri Keşfi (EDA) Başlıyor...\n")

    # 1. Veriyi Yükle
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Veri seti yüklendi. Toplam satır sayısı: {len(df)}\n")
    except FileNotFoundError:
        print(f"🚨 Hata: {DATASET_PATH} bulunamadı. Lütfen veriyi doğru klasöre koyun.")
        return

    # 2. Sınıf Dağılımı (Class Distribution)
    print("📊 1. Sınıf Dağılımı Hesaplanıyor ve Çiziliyor...")
    plt.figure(figsize=(8, 6))
    
    # Seaborn ile bar grafiği
    ax = sns.countplot(data=df, x='generated', hue='generated', palette='Set2', legend=False)
    plt.title('Yapay Zeka (1) vs İnsan (0) Metin Dağılımı', fontsize=14)
    plt.xlabel('Sınıf (0: İnsan, 1: AI)', fontsize=12)
    plt.ylabel('Metin Sayısı', fontsize=12)

    # Barların üzerine sayıları yazdır
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    # ---> YENİ EKLENEN KISIM <---
    # Yazıların grafiğin üstünden kesilmemesi için Y ekseninin tavanını %15 oranında yükseltiyoruz.
    ax.set_ylim(0, df['generated'].value_counts().max() * 1.15)
    # ----------------------------

    class_dist_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Grafik kaydedildi: {class_dist_path}\n")

    # 3. Metin Uzunluğu İstatistikleri (Text Length Statistics)
    print("📏 2. Metin Uzunlukları Analiz Ediliyor...")
    # Metinleri boşluklardan bölerek kelime sayısını bulalım
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    print("--- Sınıflara Göre Kelime Sayısı İstatistikleri ---")
    stats = df.groupby('generated')['word_count'].describe()
    print(stats)
    print("-" * 50 + "\n")

    # Histogram Çizimi
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', hue='generated', bins=50, kde=True, palette='Set2')
    plt.title('Kelime Sayısı Dağılımı (İnsan vs AI)', fontsize=14)
    plt.xlabel('Kelime Sayısı', fontsize=12)
    plt.ylabel('Frekans', fontsize=12)
    
    # Çok uzun metinler (outliers) grafiği bozmasın diye x eksenini %99'luk dilime göre kesiyoruz
    limit = df['word_count'].quantile(0.99)
    plt.xlim(0, limit)

    length_dist_path = os.path.join(FIGURES_DIR, "text_length_distribution.png")
    plt.savefig(length_dist_path, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Grafik kaydedildi: {length_dist_path}\n")

    # 4. Örnek Metinler (Example Texts)
    print("📝 3. Veri Setinden Rastgele Örnekler:\n")
    
    human_sample = df[df['generated'] == 0]['text'].sample(1, random_state=42).values[0]
    print(f"👤 İNSAN ÖRNEĞİ (Kısaltılmış):\n> {human_sample[:400]}...\n")

    ai_sample = df[df['generated'] == 1]['text'].sample(1, random_state=42).values[0]
    print(f"🤖 YAPAY ZEKA ÖRNEĞİ (Kısaltılmış):\n> {ai_sample[:400]}...\n")
    
    print("🚀 Veri Keşfi (EDA) Tamamlandı! Raporunuz için grafikler hazır.")

if __name__ == "__main__":
    run_eda()