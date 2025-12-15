import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import sys

# --- KONFIGURASI FILE ---
MODEL_FILE = 'nb_sentiment_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
LABEL_COLUMN = 'label_manual' 

def train_and_evaluate_model(input_file):
    """
    Melatih model Naive Bayes menggunakan data berlabel dari input_file,
    menghitung metrik, dan menyimpan model serta vectorizer.
    Mengembalikan dictionary metrik evaluasi.
    """
    try:
        # 1. MEMUAT DATA
        df = pd.read_csv(input_file)

        # Cek kolom yang diperlukan
        if 'clean_text' not in df.columns or LABEL_COLUMN not in df.columns:
            raise KeyError(f"Kolom 'clean_text' atau '{LABEL_COLUMN}' tidak ditemukan di file CSV.")

        df.dropna(subset=['clean_text', LABEL_COLUMN], inplace=True)
        
        X = df['clean_text'] 
        y = df[LABEL_COLUMN] 

        # Cek apakah ada cukup data untuk training/testing
        if len(df) < 20:
            print("ERROR: Data terlalu sedikit untuk diuji coba.")
            return None, None, None

        # 2. PEMISAHAN DATA (TRAINING & TESTING)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. FEATURE EXTRACTION (TF-IDF)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # 4. TRAINING MODEL
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # 5. EVALUASI MODEL
        y_pred = model.predict(X_test_tfidf)
        
        # Hitung Metrik
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Ekstrak metrik utama
        metrics = {
            'Akurasi': accuracy,
            'F1 Score (Macro)': report['macro avg']['f1-score'],
            'Presisi (Macro)': report['macro avg']['precision'],
            'Recall (Macro)': report['macro avg']['recall'],
            'Classification_Report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # 6. SIMPAN MODEL DAN VECTORIZER
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)

        return model, vectorizer, metrics

    except FileNotFoundError:
        print(f"Error: File input tidak ditemukan.")
        return None, None, None
    except KeyError as e:
        print(f"Error: Kesalahan Kolom di CSV: {e}")
        return None, None, None
    except ValueError as e:
        print(f"Error: Kesalahan data atau pembagian data: {e}")
        return None, None, None

def load_ml_assets():
    """Memuat model dan vectorizer yang sudah disimpan."""
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except FileNotFoundError:
        return None, None