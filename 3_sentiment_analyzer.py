import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Pastikan VADER sudah terinstal ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Mengunduh vader_lexicon...")
    nltk.download('vader_lexicon')

# Inisialisasi Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """Menganalisis sentimen menggunakan VADER dan mengembalikan skor compound."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0 # Kembali 0 jika teks kosong/bukan string

    # Dapatkan skor sentimen
    vs = sia.polarity_scores(text)
    
    # Skor Compound adalah ukuran sentimen keseluruhan yang dinormalisasi
    return vs['compound']

def classify_sentiment(score):
    """Mengklasifikasikan skor compound menjadi label Positif, Negatif, atau Netral."""
    if score >= 0.05:
        return 'Positif'
    elif score <= -0.05:
        return 'Negatif'
    else:
        return 'Netral'

# --- FUNGSI UTAMA ---
def run_sentiment_analysis(input_file='komentar_clean_charlie_kirk.csv', output_file='komentar_sentiment_final.csv'):
    try:
        df = pd.read_csv(input_file)
        
        # Kolom 'clean_text' adalah hasil dari preprocessing Anda
        if 'clean_text' not in df.columns:
            print("Error: Kolom 'clean_text' tidak ditemukan. Pastikan file input benar.")
            return

        print(f"Memulai analisis sentimen pada {len(df)} komentar...")
        
        # 1. Hitung Skor Compound
        df['sentiment_score'] = df['clean_text'].apply(analyze_sentiment_vader)
        
        # 2. Klasifikasikan Sentimen
        df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

        # 3. Tampilkan Ringkasan Hasil
        sentiment_counts = df['sentiment_label'].value_counts(normalize=True) * 100
        print("\n--- Ringkasan Distribusi Sentimen ---")
        print(sentiment_counts.round(2).to_string())
        
        # 4. Simpan Hasil Akhir
        df.to_csv(output_file, index=False)
        print(f"\n✅ Analisis Selesai! Data disimpan ke: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_sentiment_analysis()