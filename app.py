import streamlit as st
import pandas as pd
import os
import joblib

import matplotlib.pyplot as plt 
import seaborn as sns # (Jika belum ada)

# --- 1. IMPORT MODUL CUSTOM ---
# Pastikan file ml_trainer.py dan preprocessing.py berada di direktori yang sama
import ml_trainer 
import preprocessing 

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Analisis Sentimen Naive Bayes",
    layout="wide"
)

# Konfigurasi nama file
LABELED_DATA_FILE = 'komentar_labelled_manual.csv' # Pastikan nama file berlabel Anda sama
MODEL_FILE = 'nb_sentiment_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'


# --- 2. FUNGSI PREDIKSI BARU ---
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
    
    # Konversi label numerik ke teks
    # Menggunakan skala 1, 0, -1
    if prediction == 1:
        return "POSITIF (1)"
    elif prediction == -1:
        return "NEGATIF (-1)"
    else: # Termasuk 0
        return "NETRAL (0)"

# --- 3. MANAJEMEN SESSION STATE (Memuat Model Saat Startup) ---
# Muat model dan vectorizer yang sudah ada saat aplikasi pertama kali dijalankan
if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.session_state.model, st.session_state.vectorizer = ml_trainer.load_ml_assets()

# --- 4. TAMPILAN SIDEBAR (TEMPAT TRAINING MODEL) ---
with st.sidebar:
    st.header("⚙️ Pengaturan Model")

    st.subheader("1. Pelatihan Model")
    st.markdown(f"**Data Latih:** `{LABELED_DATA_FILE}`")

    # Tombol untuk Melatih Model
    if st.button("Latih Model ML"):
        if not os.path.exists(LABELED_DATA_FILE):
            st.error(f"File data berlabel '{LABELED_DATA_FILE}' tidak ditemukan!")
        else:
            with st.spinner("Memproses TF-IDF dan Melatih Model Naive Bayes..."):
                # Panggil fungsi training dari modul ml_trainer.py
                model, vectorizer, metrics = ml_trainer.train_and_evaluate_model(LABELED_DATA_FILE)
            
            if model and vectorizer and metrics:
                # Simpan model yang baru dilatih ke session state
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                
                st.success("✅ Pelatihan Selesai!")
                
                # Tampilkan hasil metrik
                st.subheader("Hasil Evaluasi")
                # Menggunakan DataFrame untuk menampilkan metrik utama
                df_metrics = pd.DataFrame({
                    'Metrik': list(metrics.keys())[:-1], # Kecuali Classification Report
                    'Nilai': [metrics[key] for key in list(metrics.keys())[:-1]]
                }).set_index('Metrik')
                st.dataframe(df_metrics.style.format({'Nilai': "{:.4f}"}))
                
                st.markdown("---")
                st.text("Detail Classification Report:")
                st.code(metrics['Classification_Report'])
            else:
                st.error("Pelatihan Model Gagal. Cek pesan error di terminal atau pastikan data cukup.")

    # Status Model
    st.subheader("2. Status Model Saat Ini")
    if st.session_state.model:
        st.success(f"Model ({MODEL_FILE}) siap digunakan.")
        if st.session_state.vectorizer:
            st.caption(f"Vectorizers: {len(st.session_state.vectorizer.vocabulary_)} fitur")
    else:
        st.warning("Model belum dilatih atau file model tidak ditemukan.")

# --- 5. TAMPILAN UTAMA (PREDIKSI) ---
st.title("Proyek Klasifikasi Sentimen Komentar (TF-IDF & Naive Bayes)")
st.markdown("Aplikasi untuk menganalisis sentimen menggunakan model Naive Bayes yang dilatih dengan data berlabel Anda.")

# Pengecekan Ketersediaan Model
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
            
            # Tampilkan teks setelah preprocessing
            st.caption(f"Teks setelah Preprocessing: `{preprocessing.preprocess_text(input_text)}`")
        else:
            st.error("Kolom input tidak boleh kosong.")

    st.markdown("---")

    # B. Prediksi File CSV 
    st.header("B. Prediksi Massal (CSV)")
    uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi (harus memiliki kolom 'text')", type="csv")

    if uploaded_file is not None:
        try:
            df_predict = pd.read_csv(uploaded_file)
            st.success(f"File dimuat: {len(df_predict)} baris.")
            
            if 'text' not in df_predict.columns:
                 st.error("CSV harus memiliki kolom bernama 'text'.")
            else:
                st.subheader("Memproses dan Memprediksi...")
                progress_bar = st.progress(0)
                
                # Preprocessing Massal
                df_predict['clean_text'] = df_predict['text'].apply(preprocessing.preprocess_text)
                progress_bar.progress(50)
                
                # Lakukan Prediksi Massal
                text_vectors = st.session_state.vectorizer.transform(df_predict['clean_text'])
                predictions = st.session_state.model.predict(text_vectors) # Hasilnya adalah NumPy Array

                # --- BAGIAN KRITIS YANG DIPERBAIKI ---
                label_map = {1: "POSITIF", -1: "NEGATIF", 0: "NETRAL"}
                
                # Menggunakan Pandas Series.map() untuk konversi label
                df_predict['sentimen_prediksi'] = pd.Series(predictions).astype(int).map(label_map)
                
                progress_bar.progress(100)
                
                st.subheader("Hasil Prediksi Massal")
                st.dataframe(df_predict[['text', 'sentimen_prediksi']].head(20))

                # --- BAGIAN BARU: TAMPILKAN HASIL AKHIR DAN VISUALISASI ---
                st.header("✨ Hasil Prediksi Sentimen Massal")
                
                # 1. Metrik Ringkasan Hasil
                sentiment_counts = df_predict['sentimen_prediksi'].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Komentar", len(df_predict))
                col2.metric("Positif", sentiment_counts.get("POSITIF", 0))
                col3.metric("Negatif", sentiment_counts.get("NEGATIF", 0))
                col4.metric("Netral", sentiment_counts.get("NETRAL", 0))

                # 2. Visualisasi
                st.subheader("Distribusi Sentimen (Visualisasi)")
                # Asumsi: Kita buat fungsi visualisasi sementara di sini
                # Anda akan menggunakan logika dari create_sentiment_charts,
                # tetapi harus disesuaikan untuk df_predict dan label "POSITIF"/"NEGATIF"
                
                # --- Gunakan logika sederhana untuk menampilkan Pie Chart sebagai ganti Bar Chart ---
                fig, ax = plt.subplots(figsize=(6, 6))
                labels = sentiment_counts.index
                sizes = sentiment_counts.values
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Contoh warna
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal') # Memastikan Pie Chart melingkar sempurna
                st.pyplot(fig) 
                
                # 3. Sampel Data (Hanya 20 Baris)
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