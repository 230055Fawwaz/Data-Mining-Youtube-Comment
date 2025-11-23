import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sentiment(input_file='komentar_sentiment_final.csv'):
    """Memuat data hasil analisis sentimen dan membuat visualisasi."""
    try:
        df = pd.read_csv(input_file)
        
        # Periksa apakah kolom hasil analisis sentimen ada
        if 'sentiment_label' not in df.columns:
            print("Error: Kolom 'sentiment_label' tidak ditemukan. Jalankan sentiment_analyzer.py terlebih dahulu.")
            return
        
        # --- 1. VISUALISASI DISTRIBUSI SENTIMEN (DIAGRAM BATANG) ---
        plt.figure(figsize=(8, 6))
        
        # Hitung frekuensi label
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Urutan label agar konsisten (Negatif, Netral, Positif)
        order = ['Negatif', 'Netral', 'Positif']
        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=order, ordered=True)
        sentiment_counts = sentiment_counts.sort_values('Sentiment')
        
        # Buat Bar Plot dengan warna yang sesuai
        colors = {'Negatif': 'red', 'Netral': 'gray', 'Positif': 'green'}
        bar_colors = [colors[cat] for cat in sentiment_counts['Sentiment']]
        
        ax = sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, 
                        palette=bar_colors, hue='Sentiment', legend=False)
        
        # Tambahkan nilai persentase di atas setiap batang
        total = len(df)
        for index, row in sentiment_counts.iterrows():
            count = row['Count']
            percentage = f'({(count/total)*100:.1f}%)'
            plt.text(index, count + (total * 0.02), f'{count}\n{percentage}', 
                     color='black', ha="center", fontweight='bold')
        
        plt.title('Distribusi Sentimen Netizen Komentar Kematian Charlie Kirk', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Kategori Sentimen', fontsize=12)
        plt.ylabel('Jumlah Komentar', fontsize=12)
        plt.tight_layout()
        plt.show()  # Tampilkan Grafik Batang
        
        # --- 2. VISUALISASI SKOR SENTIMEN (HISTOGRAM) ---
        plt.figure(figsize=(10, 5))
        sns.histplot(df['sentiment_score'], bins=30, kde=True, color='skyblue')
        
        # Tambahkan garis batas (threshold) sentimen VADER
        plt.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Batas Positif (0.05)')
        plt.axvline(x=-0.05, color='red', linestyle='--', linewidth=2, label='Batas Negatif (-0.05)')
        
        plt.title('Histogram Distribusi Skor Sentimen Compound', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Skor Sentimen Compound (VADER)', fontsize=12)
        plt.ylabel('Frekuensi', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()  # Tampilkan Histogram
        
        # Cetak statistik ringkasan
        print("\n=== RINGKASAN ANALISIS SENTIMEN ===")
        print(f"Total komentar: {total}")
        print("\nDistribusi Sentimen:")
        for _, row in sentiment_counts.iterrows():
            pct = (row['Count']/total)*100
            print(f"  {row['Sentiment']}: {row['Count']} ({pct:.1f}%)")
        print(f"\nSkor rata-rata: {df['sentiment_score'].mean():.3f}")
        print(f"Skor median: {df['sentiment_score'].median():.3f}")
        
    except FileNotFoundError:
        print(f"Error: File input tidak ditemukan. Pastikan file '{input_file}' ada.")
    except Exception as e:
        print(f"Terjadi kesalahan saat visualisasi: {e}")

# --- Eksekusi Program ---
if __name__ == "__main__":
    visualize_sentiment()