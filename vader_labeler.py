import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- KONFIGURASI FILE OUTPUT ---
MODEL_FILE = 'nb_sentiment_vader_model.pkl'
VECTORIZER_FILE = 'tfidf_vader_vectorizer.pkl'

# --- INISIALISASI VADER ---
# Kita asumsikan ini sudah diinisialisasi oleh app.py, tapi kita inisialisasi lagi di sini
# untuk memastikan SIA tersedia jika diimport.
try:
    SIA = SentimentIntensityAnalyzer()
except:
    # Jika SIA gagal, model training akan gagal.
    SIA = None 


def vader_label_and_train(df_input, preprocess_func):
    """
    Memberi label otomatis menggunakan VADER, lalu melatih model ML
    menggunakan fungsi preprocess_func yang di-pass dari app.py.
    """
    if SIA is None:
        return None, None, {"Error": "VADER Lexicon tidak terinisialisasi. Cek NLTK download."}
        
    try:
        # 1. Pre-processing (Menggunakan fungsi yang di-pass dari app.py)
        df_input['clean_text'] = df_input['text'].apply(preprocess_func)

        # 2. Pelabelan Otomatis (VADER)
        df_input['sentiment_score'] = df_input['clean_text'].apply(lambda x: SIA.polarity_scores(x)['compound'])
        
        # Konversi VADER Compound Score ke Label (-1, 0, 1)
        def vader_to_label(score):
            if score >= 0.05:
                return 1  # Positif
            elif score <= -0.05:
                return -1 # Negatif
            else:
                return 0  # Netral

        df_input['vader_label'] = df_input['sentiment_score'].apply(vader_to_label)
        
        # --- TRAINING ML DENGAN LABEL VADER ---
        df = df_input.dropna(subset=['clean_text', 'vader_label'])
        
        X = df['clean_text'] 
        y = df['vader_label'] 

        if len(df) < 50:
             return None, None, {"Error": "Data terlalu sedikit setelah pelabelan VADER. Minimal 50 baris."}

        # Pemisahan Data & TF-IDF & Training Model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Evaluasi
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            'Akurasi': accuracy_score(y_test, y_pred),
            'F1 Score (Macro)': report['macro avg']['f1-score'],
            'Presisi (Macro)': report['macro avg']['precision'],
            'Recall (Macro)': report['macro avg']['recall'],
            'Classification_Report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # Simpan Model dan Vectorizer
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)

        return model, vectorizer, metrics

    except FileNotFoundError:
        return None, None, {"Error": f"File input tidak ditemukan."}
    except Exception as e:
        return None, None, {"Error": f"Kesalahan training ML: {e}"}

def load_ml_assets():
    """Memuat model dan vectorizer yang sudah disimpan."""
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except FileNotFoundError:
        return None, None