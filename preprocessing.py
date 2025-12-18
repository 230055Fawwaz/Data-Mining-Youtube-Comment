import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- PENANGANAN DOWNLOAD NLTK ---
# Menggunakan LookupError untuk NLTK v3.x ke atas
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# --- CUSTOM STOPWORDS ---
# Daftar kata yang TIDAK boleh dihapus (negasi, yang penting untuk sentimen)
negation_words = set(['not', 'no', 'never', 'n\'t']) 
english_stop_words = set(stopwords.words('english')).difference(negation_words)


# Inisialisasi Lemmatizer dan Stemmer di luar fungsi untuk efisiensi
lemmatizer = WordNetLemmatizer()
factory = StemmerFactory()
id_stemmer = factory.create_stemmer() # Jaga-jaga jika ada Bahasa Indonesia

# --- FUNGSI PRE-PROCESSING UTAMA ---
def preprocess_text(text):
    # Cek jika input bukan string atau kosong
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # 1. Penanganan Data Mentah (Hapus karakter di awal/akhir)
    text = text.strip().strip('"') 

    # 2. Case Folding (lowercase semua)
    text = text.lower()
    
    # 3. Penanganan Karakter Khusus, Non-ASCII, dan Unicode Rusak
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 4. Hapus Username (@) dan Teks Angka
    text = re.sub(r'@\S+', '', text) 
    text = re.sub(r'\d+', '', text) 
    
    # 5. Hapus Punctuation & Simbol
    # Hanya pertahankan huruf, spasi, dan tanda penting
    text = re.sub(r'[^a-z\s]', ' ', text) 
    
    # Hapus spasi berulang
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenizing
    tokens = word_tokenize(text)
    
    # 7. Stopword Removal (Menggunakan custom_stopwords di atas yang menjaga negasi)
    tokens = [word for word in tokens if word not in english_stop_words and len(word) > 2]
    
    # 8. Lemmatization (Mengurangi kata ke bentuk dasar yang valid)
    # Gunakan pos='v' (verb) agar lebih efektif mengubah bentuk V2/V3 ke V1
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    # Mengembalikan tokens yang SUDAH dilemmatisasi
    return " ".join(tokens)

# --- FUNGSI UTAMA UNTUK MENGOLAH FILE ---
def run_preprocessing(input_file='komentar_charlie_kirk_death.csv', output_file='komentar_clean_charlie_kirk.csv'):
    try:
        df = pd.read_csv(input_file)
        
        try:
             if 'text,author,published_at,like_count' in df.columns:
                 df[['text', 'author', 'published_at', 'like_count']] = df['text,author,published_at,like_count'].str.split(',', expand=True, n=3)
                 df = df.drop(columns=['text,author,published_at,like_count'])
                 print("Kolom berhasil dipisahkan kembali.")
        except:
             pass 
        
        if 'text' not in df.columns:
             print("Error: Kolom 'text' tidak ditemukan. Pastikan format CSV benar.")
             return

        print(f"Memulai pembersihan {len(df)} baris komentar...")
        df['clean_text'] = df['text'].apply(preprocess_text)
        
        print("\n--- 5 Komentar Awal Setelah Pembersihan ---")
        print(df[['text', 'clean_text']].head())
        
        df.to_csv(output_file, index=False)
        print(f"\n✅ Proses Selesai! Data telah disimpan ke: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan. Pastikan nama file dan lokasinya benar.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# --- Eksekusi Program ---
if __name__ == "__main__":
    run_preprocessing(input_file='komentar_charlie_kirk_death.csv', output_file='komentar_clean_charlie_kirk.csv')
    print("Data bersih telah disimpan ke 'komentar_clean_charlie_kirk.csv' dalam format CSV.")