import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns # Pastikan library ini diimport
import sys # Digunakan untuk error reporting jika diperlukan

# --- 1. IMPORT MODUL CUSTOM ---
# Pastikan file ml_trainer.py dan preprocessing.py berada di direktori yang sama
try:
    import ml_trainer 
    import preprocessing 
except ImportError as e:
    st.error(f"Error saat mengimpor modul kustom: {e}. Pastikan ml_trainer.py dan preprocessing.py ada.")


# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Analisis Sentimen Naive Bayes",
    layout="wide"
)

# --- KONFIGURASI FILE HARDCODED ---
LABELED_DATA_FILE = 'komentar_labelled_manual.csv' # File untuk TRAINING (Sudah hardcoded)
PREDICTION_INPUT_FILE = 'komentar_clean_charlie_kirk.csv' # File untuk PREDIKSI MASSAL (BARU di hardcode)
MODEL_FILE = 'nb_sentiment_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'


# --- 2. FUNGSI PREDIKSI DAN VISUALISASI ---
def predict_sentiment(text, model, vectorizer):
    """Membersihkan teks dan memprediksi sentimen."""
    if not model or not vectorizer:
        return "Model Belum Tersedia. Silakan Latih Model Terlebih Dahulu."

    # Preprocessing teks baru menggunakan fungsi dari preprocessing.py
    clean_text = preprocessing.preprocess_text(text)
    
    # Jika teks bersih kosong, anggap Netral
    if not clean_text:
        return "NETRAL (0)"

    # Transformasi teks menggunakan Vectorizer yang sudah dilatih
    text_vector = vectorizer.transform([clean_text])

    # Prediksi
    prediction = model.predict(text_vector)[0]
    
    # --- PERBAIKAN: MAPPING 5-KELAS UNTUK PREDIKSI TUNGGAL ---
    label_map = {
        2: "SANGAT POSITIF (2)", 
        1: "POSITIF (1)", 
        0: "NETRAL (0)", 
        -1: "NEGATIF (-1)", 
        -2: "SANGAT NEGATIF (-2)"
    }
    
    return label_map.get(prediction, "LABEL TIDAK DIKENALI")

def create_sentiment_chart(df):
    """Membuat dan menampilkan Pie Chart distribusi sentimen."""
    sentiment_counts = df['sentimen_prediksi'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    # Tentukan warna yang konsisten
    color_map = {"POSITIF": "#32CD32", "NEGATIF": "#FF4500", "NETRAL": "#ADD8E6"}
    colors = [color_map.get(label, "#CCCCCC") for label in labels]
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal') 
    st.pyplot(fig)


# --- 3. MANAJEMEN SESSION STATE (Memuat Model Saat Startup) ---
if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.session_state.model, st.session_state.vectorizer = ml_trainer.load_ml_assets()

# --- 4. TAMPILAN SIDEBAR (TEMPAT TRAINING MODEL) ---
with st.sidebar:
    st.header("⚙️ Pengaturan Model")

    st.subheader("1. Pelatihan Model")
    st.markdown(f"**Data Latih:** `{LABELED_DATA_FILE}`")

    if st.button("Latih Model ML"):
        if not os.path.exists(LABELED_DATA_FILE):
            st.error(f"File data berlabel '{LABELED_DATA_FILE}' tidak ditemukan!")
        else:
            with st.spinner("Memproses TF-IDF dan Melatih Model Naive Bayes..."):
                model, vectorizer, metrics = ml_trainer.train_and_evaluate_model(LABELED_DATA_FILE)
            
            if model and vectorizer and metrics:
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
            else:
                st.error("Pelatihan Model Gagal. Cek terminal untuk detail error.")


    st.subheader("2. Status Model Saat Ini")
    if st.session_state.model:
        st.success(f"Model ({MODEL_FILE}) siap digunakan.")
    else:
        st.warning("Model belum dilatih atau file model tidak ditemukan.")

# --- 5. TAMPILAN UTAMA (PREDIKSI) ---
st.title("Proyek Klasifikasi Sentimen Komentar (TF-IDF & Naive Bayes)")
st.markdown("Aplikasi untuk menganalisis sentimen menggunakan model Naive Bayes yang dilatih dengan data berlabel Anda.")

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
            st.caption(f"Teks setelah Preprocessing: `{preprocessing.preprocess_text(input_text)}`")
        else:
            st.error("Kolom input tidak boleh kosong.")

    st.markdown("---")

    # B. Prediksi Massal (HARDCODED FILE)
    st.header("B. Prediksi Massal (File Hardcoded)")
    st.markdown(f"Prediksi akan dilakukan pada file: **`{PREDICTION_INPUT_FILE}`**")
    
    if st.button("Jalankan Prediksi Massal"):
        if not os.path.exists(PREDICTION_INPUT_FILE):
             st.error(f"Error: File input untuk prediksi '{PREDICTION_INPUT_FILE}' tidak ditemukan. Pastikan file sudah ada.")
        else:
            try:
                df_predict = pd.read_csv(PREDICTION_INPUT_FILE)
                st.success(f"File dimuat: {len(df_predict)} baris.")
                
                if 'text' not in df_predict.columns:
                     st.error("File harus memiliki kolom bernama 'text'.")
                else:
                    st.subheader("Memproses dan Memprediksi...")
                    progress_bar = st.progress(0)
                    
                    # Preprocessing Massal
                    df_predict['clean_text'] = df_predict['text'].apply(preprocessing.preprocess_text)
                    progress_bar.progress(50)
                    
                    # Lakukan Prediksi Massal
                    text_vectors = st.session_state.vectorizer.transform(df_predict['clean_text'])
                    predictions = st.session_state.model.predict(text_vectors)
                    
                    # Konversi hasil prediksi ke label teks (menggunakan map)
                    label_map_5_class = {
                        2: "SANGAT POSITIF", 
                        1: "POSITIF", 
                        0: "NETRAL", 
                        -1: "NEGATIF", 
                        -2: "SANGAT NEGATIF"
                    }
                    df_predict['sentimen_prediksi'] = pd.Series(predictions).astype(int).map(label_map_5_class)
                    
                    progress_bar.progress(100)
                    
                    # TAMPILAN HASIL AKHIR DAN VISUALISASI
                    st.header("✨ Hasil Prediksi Sentimen Massal")
                    sentiment_counts = df_predict['sentimen_prediksi'].value_counts()
    
                    # Ganti 4 kolom menjadi 5 atau 6 kolom untuk menampilkan -2 dan +2
                    col_neg2, col_neg1, col_netral, col_pos1, col_pos2 = st.columns(5)
                    
                    col_neg2.metric("S. Negatif (-2)", sentiment_counts.get("SANGAT NEGATIF", 0))
                    col_neg1.metric("Negatif (-1)", sentiment_counts.get("NEGATIF", 0))
                    col_netral.metric("Netral (0)", sentiment_counts.get("NETRAL", 0))
                    col_pos1.metric("Positif (+1)", sentiment_counts.get("POSITIF", 0))
                    col_pos2.metric("S. Positif (+2)", sentiment_counts.get("SANGAT POSITIF", 0))

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
                st.exception(e) # Menampilkan detail traceback untuk debug