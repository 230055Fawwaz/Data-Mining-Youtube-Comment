import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns # Diperlukan untuk Bar Chart
import sys
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

# --- IMPORT MODUL CUSTOM (vader_labeler.py HARUS ADA DI DIREKTORI YANG SAMA) ---
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
PREDICTION_INPUT_FILE = 'komentar_charlie_kirk_death' 

# Fungsi Prediksi
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

# --- FUNGSI VISUALISASI GRAFIK ---

# Fungsi 1: Pie Chart (Persentase NB)
def create_nb_pie_chart(df):
    """Membuat dan menampilkan Pie Chart distribusi sentimen prediksi NB."""
    sentiment_counts = df['sentimen_prediksi'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6)) # Ukuran Pie Chart
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    
    color_map = {"POSITIF": "#32CD32", "NEGATIF": "#FF4500", "NETRAL": "#ADD8E6"}
    colors = [color_map.get(label.split()[0], "#CCCCCC") for label in labels] 
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    ax.axis('equal') 
    ax.set_title('Persentase Sentimen Naive Bayes')
    return fig

# Fungsi 2: Bar Chart NB (Hitungan)
def create_nb_bar_chart(df):
    """Membuat Bar Chart Sentimen Prediksi NB (Fig. 2 Style)."""
    sentiment_counts = df['sentimen_prediksi'].value_counts().reindex(['NEGATIF', 'NETRAL', 'POSITIF'], fill_value=0)
    total = sentiment_counts.sum()
    
    fig, ax = plt.subplots(figsize=(8, 5)) 
    
    # Warna sesuai Fig. 2 (Merah Bata, Abu-abu, Hijau Tua)
    color_map = {'POSITIF': '#2ca02c', 'NETRAL': '#949494', 'NEGATIF': '#d62728'} 
    colors = [color_map[label] for label in sentiment_counts.index]
    
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=colors)
    
    ax.set_title('Sentiment Distribution of YouTube Comments (NB Predicted)')
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Sentiment Category')
    
    # Tambahkan nilai (Count dan Persentase) di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        percentage = f'({(height/total)*100:.1f}%)'
        ax.annotate(f'{int(height)}\n{percentage}', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   fontsize=10, fontweight='bold')
        
    ax.set_ylim(0, max(sentiment_counts.values) * 1.15) 

    return fig

# Fungsi 3: Histogram VADER (Distribusi Score)
def create_vader_histogram(df):
    """Membuat Histogram Distribusi Compound Score VADER (Fig. 3 Style)."""
    
    if 'vader_score' not in df.columns:
        # Ini seharusnya sudah dihitung di prediksi massal, tapi sebagai fallback
        SIA = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        df['vader_score'] = df['clean_text'].apply(lambda x: SIA.polarity_scores(x)['compound'])
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Gunakan sns.histplot dengan KDE untuk mereplikasi tampilan (Fig. 3)
    sns.histplot(df['vader_score'], bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black')
    
    # Tambahkan garis threshold
    ax.axvline(x=0.05, color='green', linestyle='--', label='Positive Threshold (0.05)')
    ax.axvline(x=-0.05, color='red', linestyle='--', label='Negative Threshold (-0.05)')
    
    ax.set_title('Distribution of VADER Compound Sentiment Scores')
    ax.set_xlabel('Compound Score')
    ax.set_ylabel('Frequency') 
    ax.legend()
    ax.set_xlim(-1.0, 1.0) 

    return fig


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
                model, vectorizer, metrics = vader_labeler.vader_label_and_train(
                    df_raw, 
                    preprocess_text 
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
                    progress_bar.progress(30)
                    
                    # 1. Lakukan Prediksi Naive Bayes
                    text_vectors = st.session_state.vectorizer.transform(df_predict['clean_text'])
                    predictions = st.session_state.model.predict(text_vectors)
                    
                    label_map_3_class = {1: "POSITIF", 0: "NETRAL", -1: "NEGATIF"}
                    df_predict['sentimen_prediksi'] = pd.Series(predictions).astype(int).map(label_map_3_class)
                    
                    progress_bar.progress(60)

                    # 2. HITUNG VADER COMPOUND SCORE UNTUK CHART KETIGA
                    df_predict['vader_score'] = df_predict['clean_text'].apply(lambda x: SIA.polarity_scores(x)['compound'])
                    
                    progress_bar.progress(80)
                    
                    # --- TAMPILAN HASIL AKHIR DAN VISUALISASI ---
                    st.header("✨ Hasil Prediksi Sentimen Massal")
                    sentiment_counts = df_predict['sentimen_prediksi'].value_counts()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Komentar", len(df_predict))
                    col2.metric("Positif (NB)", sentiment_counts.get("POSITIF", 0))
                    col3.metric("Negatif (NB)", sentiment_counts.get("NEGATIF", 0))
                    col4.metric("Netral (NB)", sentiment_counts.get("NETRAL", 0))

                    st.subheader("Visualisasi Sentimen")
                    
                    # SUSUNAN GRAFIK: PIE dan BAR NB berdampingan
                    col_pie, col_bar_nb = st.columns([1, 2])
                    
                    with col_pie:
                        st.markdown("##### 1. Ringkasan Persentase (NB Predicted)")
                        fig_pie = create_nb_pie_chart(df_predict)
                        st.pyplot(fig_pie)
                        
                    with col_bar_nb:
                        st.markdown("##### 2. Distribusi Hitungan (NB Predicted)")
                        fig_bar_nb = create_nb_bar_chart(df_predict)
                        st.pyplot(fig_bar_nb)
                    
                    st.markdown("---")
                    
                    # --- CHART 3: DISTRIBUSI VADER COMPOUND SCORE ---
                    st.markdown("##### 3. Distribusi VADER Compound Score (Linguistik)")
                    
                    # Plot Histogram VADER (Mereplikasi Fig. 3)
                    fig_vader_hist = create_vader_histogram(df_predict)
                    st.pyplot(fig_vader_hist)

                    progress_bar.progress(100)
                    
                    st.subheader("Sampel Data (20 Baris Teratas)")
                    st.dataframe(df_predict[['text', 'sentimen_prediksi', 'vader_score']].head(20))
                    
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