import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Untuk jaga-jaga jika ada teks Bhs Indonesia

# --- Download NLTK resources (Jalankan sekali jika belum pernah) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Daftar Stopwords Bahasa Inggris
from nltk.corpus import stopwords
english_stop_words = set(stopwords.words('english'))

# Inisialisasi Sastrawi Stemmer (Untuk teks Bhs Indonesia)
factory = StemmerFactory()
id_stemmer = factory.create_stemmer()

# --- FUNGSI PRE-PROCESSING UTAMA ---
def preprocess_text(text):
    # 1. Penanganan Data Mentah (Jika Ada Karakter di Awal Baris)
    # Hapus karakter non-teks di awal/akhir kutipan
    text = text.strip().strip('"') 

    # 2. Case Folding
    text = text.lower()
    
    # 3. Penanganan Karakter Khusus, Non-ASCII, dan Unicode Emoji
    # Hapus karakter unicode yang rusak (seperti dari contoh: ðŸŽ‰, ðŸ™„, â¤)
    # Ini menangani encoding yang rusak dari database atau CSV
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 4. Hapus Username (@) dan Teks Angka
    text = re.sub(r'@\S+', '', text) # Hapus @username
    text = re.sub(r'\d+', '', text) # Hapus Angka
    
    # 5. Hapus Punctuation, Simbol, dan Spasi Berlebih
    # Hanya pertahankan huruf, spasi, dan tanda penting (opsional)
    text = re.sub(r'[^a-z\s]', ' ', text) 
    
    # Hapus spasi berulang
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenizing
    tokens = word_tokenize(text)
    
    # 7. Stopword Removal (Filter kata yang terlalu pendek atau stopword)
    tokens = [word for word in tokens if word not in english_stop_words and len(word) > 2]
    
    # 8. Opsional: Stemming/Lemmatization (Mengurangi kata ke bentuk dasar)
    # Untuk analisis yang lebih canggih, Anda bisa menggunakan NLTK Lemmatizer atau Stemmer.
    # Contoh: 'killing' -> 'kill', 'changed' -> 'chang'
    # from nltk.stem import PorterStemmer
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# --- FUNGSI UTAMA UNTUK MENGOLAH FILE ---
def run_preprocessing(input_file='komentar_charlie_kirk_death.csv', output_file='komentar_clean_charlie_kirk.csv'):
    """Memuat data CSV, membersihkan kolom 'text', dan menyimpannya ke file baru."""
    try:
        # Memuat data
        df = pd.read_csv(input_file)
        
        # Penamaan ulang kolom yang benar jika kolom tergabung (berdasarkan data contoh)
        # Jika file CSV Anda sudah benar, lewati blok try/except ini:
        try:
             # Coba asumsikan header pertama adalah gabungan, dan pisahkan kembali
             if 'text,author,published_at,like_count' in df.columns:
                 df[['text', 'author', 'published_at', 'like_count']] = df['text,author,published_at,like_count'].str.split(',', expand=True, n=3)
                 df = df.drop(columns=['text,author,published_at,like_count'])
                 print("Kolom berhasil dipisahkan kembali.")
        except:
             pass # Lewati jika sudah ada kolom yang benar
        
        # Pastikan kolom 'text' ada dan bukan NaN
        if 'text' not in df.columns:
             print("Error: Kolom 'text' tidak ditemukan. Pastikan format CSV benar.")
             return

        # Terapkan fungsi preprocessing ke setiap baris kolom 'text'
        print(f"Memulai pembersihan {len(df)} baris komentar...")
        df['clean_text'] = df['text'].apply(preprocess_text)
        
        # Opsional: Tampilkan beberapa hasil untuk verifikasi
        print("\n--- 5 Komentar Awal Setelah Pembersihan ---")
        print(df[['text', 'clean_text']].head())
        
        # Simpan DataFrame yang sudah diproses
        df.to_csv(output_file, index=False)
        print(f"\n✅ Proses Selesai! Data telah disimpan ke: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan. Pastikan nama file dan lokasinya benar.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# --- Eksekusi Program ---
if __name__ == "__main__":
    # GANTI 'input_file.csv' DENGAN NAMA FILE DARI YOUTUBE SCRAPER ANDA
    run_preprocessing(input_file='komentar_charlie_kirk_death.csv', output_file='komentar_clean_charlie_kirk.csv')