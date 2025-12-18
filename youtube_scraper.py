import googleapiclient.discovery
import pandas as pd
import sys

# --- Konfigurasi Yotube ---
API_KEY = "" 
VIDEO_ID = "9WquMr6mBCs"
MAX_COMMENTS = 500 

# --- Link Video Yotube (kalau mau) ---
# https://www.youtube.com/watch?v=9WquMr6mBCs

# Inisialisasi Layanan YouTube API
try:
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=API_KEY
    )
except Exception as e:
    print(f"Gagal menginisialisasi API. Pastikan Kunci API benar. Error: {e}")
    sys.exit()

comments_list = []
next_page_token = None

print(f"Memulai pengambilan data komentar untuk Video ID: {VIDEO_ID}...")

while len(comments_list) < MAX_COMMENTS:
    try:
        # Panggil API untuk Daftar Komentar
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=VIDEO_ID,
            # Ambil maksimal 100 per panggilan, atau sisa yang dibutuhkan
            maxResults=min(100, MAX_COMMENTS - len(comments_list)), 
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        # Ekstraksi Data dari respons
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            
            comments_list.append({
                "text": comment["textDisplay"],
                "author": comment["authorDisplayName"],
                "published_at": comment["publishedAt"],
                "like_count": comment["likeCount"]
            })
            
            if len(comments_list) >= MAX_COMMENTS:
                break

        # Pindah ke Halaman Berikutnya (jika ada)
        next_page_token = response.get("nextPageToken")
        
        if not next_page_token:
            print("Semua komentar telah diambil, atau tidak ada halaman lagi.")
            break
            
        print(f"Diambil: {len(comments_list)} komentar. Lanjut ke halaman berikutnya...")
    
    except googleapiclient.errors.HttpError as error:
        print(f"\n[ERROR API]: Terjadi kesalahan HTTP saat mengambil data.")
        if error.resp.status == 403:
            print("Pesan: Kuota API Harian Habis atau Kunci API Anda Tidak Sah.")
        else:
            print(f"Detail: {error}")
        break
    except Exception as e:
        print(f"\n[ERROR Umum]: Terjadi kesalahan tak terduga: {e}")
        break

# 3. Simpan ke DataFrame dan Ekspor
df_comments = pd.DataFrame(comments_list)
df_final = df_comments.head(MAX_COMMENTS) # Memastikan tidak melebihi 500 baris

print(f"\nSelesai! Berhasil mengambil {len(df_final)} komentar.")

# Simpan ke file CSV
df_final.to_csv('komentar_charlie_kirk_death.csv', index=False, encoding='utf-8')
print("Data telah disimpan ke 'komentar_charlie_kirk_death.csv' dalam format CSV.")