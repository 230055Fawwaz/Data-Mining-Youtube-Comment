import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt 
import sys
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Diperlukan untuk setup SIA

# --- IMPORT MODUL CUSTOM (BARU) ---
try:
    import vader_labeler
except ImportError as e:
    st.error(f"Error saat mengimpor modul 'vader_labeler.py': {e}. Pastikan file ada.")
    st.stop()


# --- 1. SETUP LINGKUNGAN NLTK & PRE-PROCESSING ---
@st.cache_resource
def setup_nltk_and_vader():
    """Mengunduh resource NLTK dan menyiapkan VADER/SIA."""
    resources = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                return f"Gagal mengunduh {resource}: {e}", None
    
    try:
        # Kita hanya perlu inisialisasi VADER di sini untuk memastikan download selesai
        return None, SentimentIntensityAnalyzer()
    except Exception as e:
        return f"Gagal inisialisasi VADER: {e}", None

error_msg, SIA = setup_nltk_and_vader()
if error_msg:
    st.error(error_msg)
    st.stop()

# Inisialisasi Lemmatizer dan Stopwords Negasi
lemmatizer = WordNetLemmatizer()
negation_words = set(['not', 'no', 'never', 'n\'t']) 
ENGLISH_STOP_WORDS = set(stopwords.words('english')).difference(negation_words)

# Fungsi Pre-processing
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.strip().strip('"').lower()
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'@\S+', '', text) 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^a-z\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    return " ".join(tokens)


# --- 2. KONFIGURASI FILE DAN FUNGSI PREDIKSI/VISUALISASI ---
MODEL_FILE = 'nb_sentiment_vader_model.pkl'
VECTORIZER_FILE = 'tfidf_vader_vectorizer.pkl'
PREDICTION_INPUT_FILE = 'komentar_charlie_kirk_death.csv' 

def predict_sentiment(text, model, vectorizer):
    if not model or not vectorizer:
        return "Model Belum Tersedia. Silakan Latih Model Terlebih Dahulu."

    clean_text = preprocess_text(text)
    if not clean_text:
        return "NETRAL (0)"

    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)[0]
    
    label_map = {
        1: "POSITIF (1)", 
        0: "NETRAL (0)", 
        -1: "NEGATIF (-1)" 
    }
    
    return label_map.get(prediction, "LABEL TIDAK DIKENALI")

def create_sentiment_chart(df):
    """Membuat dan menampilkan Pie Chart distribusi sentimen."""
    sentiment_counts = df['sentimen_prediksi'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    
    color_map = {"POSITIF": "#32CD32", "NEGATIF": "#FF4500", "NETRAL": "#ADD8E6"}
    colors = [color_map.get(label.split()[0], "#CCCCCC") for label in labels]
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal') 
    st.pyplot(fig) 


# --- 3. MANAJEMEN SESSION STATE & TAMPILAN ---
st.set_page_config(
    page_title="Analisis Sentimen VADER-Based ML",
    layout="wide"
)

# Muat model menggunakan fungsi dari vader_labeler.py
if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.session_state.model, st.session_state.vectorizer = vader_labeler.load_ml_assets()

# --- TAMPILAN SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan Model (VADER-Based)")

    st.subheader("1. Pelatihan Otomatis (VADER)")
    st.markdown(f"**Data Input:** `{PREDICTION_INPUT_FILE}`")

    if st.button("Latih Model ML (Otomatis VADER)"):
        if not os.path.exists(PREDICTION_INPUT_FILE):
            st.error(f"File input '{PREDICTION_INPUT_FILE}' tidak ditemukan!")
        else:
            df_raw = pd.read_csv(PREDICTION_INPUT_FILE)
            
            with st.spinner("Memproses VADER Labeling, TF-IDF, dan Melatih Model Naive Bayes..."):
                # PENTING: Panggil fungsi training dari modul terpisah
                model, vectorizer, metrics = vader_labeler.vader_label_and_train(
                    df_raw, 
                    preprocess_text # Pass fungsi preprocessing lokal
                )
            
            if model and vectorizer and 'Error' not in metrics:
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                
                st.success("✅ Pelatihan Selesai!")
                
                st.subheader("Hasil Evaluasi")
                df_metrics = pd.DataFrame({
                    'Metrik': list(metrics.keys())[:-1], 
                    'Nilai': [metrics[key] for key in list(metrics.keys())[:-1]]
                }).set_index('Metrik')
                st.dataframe(df_metrics.style.format({'Nilai': "{:.4f}"}))
                
                st.markdown("---")
                st.text("Detail Classification Report:")
                st.code(metrics['Classification_Report'])
            elif 'Error' in metrics:
                st.error(f"Pelatihan Gagal: {metrics['Error']}")
            else:
                st.error("Pelatihan Model Gagal. Cek terminal untuk detail error.")


    st.subheader("2. Status Model Saat Ini")
    if st.session_state.model:
        st.success(f"Model ({MODEL_FILE}) siap digunakan.")
    else:
        st.warning("Model belum dilatih atau file model tidak ditemukan.")


# --- TAMPILAN UTAMA (PREDIKSI) ---
# ... (Logika Prediksi Teks Tunggal dan Massal tetap sama seperti sebelumnya) ...
st.title("Proyek Klasifikasi Sentimen VADER-Based ML")
st.markdown("Model Naive Bayes dilatih secara otomatis menggunakan label VADER untuk prediksi.")

if not st.session_state.model:
    st.error("Model belum dilatih atau tidak ditemukan. Mohon latih model di **sidebar** terlebih dahulu.")
else:
    # A. Prediksi Teks Tunggal
    st.header("A. Prediksi Sentimen Teks Tunggal")
    input_text = st.text_area("Masukkan Komentar untuk Prediksi:", "van jones cant beat anyone debate", height=100)

    if st.button("Prediksi Sentimen"):
        if input_text:
            hasil_prediksi = predict_sentiment(
                input_text, 
                st.session_state.model, 
                st.session_state.vectorizer
            )
            st.info(f"Hasil Prediksi: **{hasil_prediksi}**")
            st.caption(f"Teks setelah Preprocessing: `{preprocess_text(input_text)}`")
        else:
            st.error("Kolom input tidak boleh kosong.")

    st.markdown("---")

    # B. Prediksi Massal (HARDCODED FILE)
    st.header("B. Prediksi Massal (File Hardcoded)")
    st.markdown(f"Prediksi akan dilakukan pada file: **`{PREDICTION_INPUT_FILE}`**")
    
    if st.button("Jalankan Prediksi Massal"):
        if not os.path.exists(PREDICTION_INPUT_FILE):
             st.error(f"Error: File input untuk prediksi '{PREDICTION_INPUT_FILE}' tidak ditemukan.")
        else:
            try:
                df_predict = pd.read_csv(PREDICTION_INPUT_FILE)
                st.success(f"File dimuat: {len(df_predict)} baris.")
                
                if 'text' not in df_predict.columns:
                     st.error("File harus memiliki kolom bernama 'text'.")
                else:
                    st.subheader("Memproses dan Memprediksi...")
                    progress_bar = st.progress(0)
                    
                    df_predict['clean_text'] = df_predict['text'].apply(preprocess_text)
                    progress_bar.progress(50)
                    
                    text_vectors = st.session_state.vectorizer.transform(df_predict['clean_text'])
                    predictions = st.session_state.model.predict(text_vectors)
                    
                    # Konversi hasil prediksi ke label teks (3 kelas VADER)
                    label_map_3_class = {
                        1: "POSITIF", 
                        0: "NETRAL", 
                        -1: "NEGATIF" 
                    }
                    df_predict['sentimen_prediksi'] = pd.Series(predictions).astype(int).map(label_map_3_class)
                    
                    progress_bar.progress(100)
                    
                    # TAMPILAN HASIL AKHIR DAN VISUALISASI
                    st.header("✨ Hasil Prediksi Sentimen Massal")
                    sentiment_counts = df_predict['sentimen_prediksi'].value_counts()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Komentar", len(df_predict))
                    col2.metric("Positif", sentiment_counts.get("POSITIF", 0))
                    col3.metric("Negatif", sentiment_counts.get("NEGATIF", 0))
                    col4.metric("Netral", sentiment_counts.get("NETRAL", 0))

                    st.subheader("Distribusi Sentimen (Visualisasi)")
                    create_sentiment_chart(df_predict)
                    
                    st.subheader("Sampel Data (20 Baris Teratas)")
                    st.dataframe(df_predict[['text', 'sentimen_prediksi']].head(20))
                    
                    # Tombol Download
                    csv_output = df_predict.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi CSV",
                        data=csv_output,
                        file_name='hasil_prediksi_sentimen.csv',
                        mime='text/csv',
                    )
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
                st.exception(e)