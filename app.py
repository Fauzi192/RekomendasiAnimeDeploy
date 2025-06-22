import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Konfigurasi halaman
st.set_page_config(page_title="🎥 Rekomendasi Anime", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating", "members"])
    df = df[df["rating"] >= 1]
    df = df.drop_duplicates(subset=["name", "genre"])
    df = df.reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()
    return df

anime_df = load_data()

# Bangun model
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix, tfidf

knn_model, tfidf_matrix, tfidf_vectorizer = build_model(anime_df)

# CSS kustom
st.markdown("""
<style>
    .anime-card {
        background-color: #fffafc;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 16px;
        border-left: 5px solid #f04e7c;
        box-shadow: 0 4px 12px rgba(240, 78, 124, 0.1);
    }
    .anime-header {
        font-size: 20px;
        font-weight: bold;
        color: #f04e7c;
        margin-bottom: 8px;
    }
    .anime-body {
        font-size: 15px;
        color: #333;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigasi
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi", "📂 Genre"])

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "🏠 Home":
    st.title("🎌 Rekomendasi Anime Favorit")

    st.markdown("""
Selamat datang di website **Rekomendasi Anime Favorit**! 🎉

Website ini dirancang khusus untuk membantu para pecinta anime dalam menemukan tontonan baru yang sesuai dengan preferensi mereka. Dengan teknologi **Content-Based Filtering** , **Term Frequency–Inverse Document Frequency (TF-IDF)**, dan algoritma **K-Nearest Neighbors (KNN)**, sistem kami akan memberikan rekomendasi anime yang mirip berdasarkan genre favoritmu.

---

### 💡 Mengapa Memilih Website Ini?

Menonton anime tidak hanya hiburan biasa, tapi juga pengalaman emosional dan estetika. Kami menyediakan alat cerdas untuk menemukan judul-judul anime yang sejalan dengan seleramu — baik itu action epik, romansa menyentuh, atau petualangan fantasi.

---

### ⚙️ Teknologi di Balik Layar

Sistem ini menggunakan pendekatan gabungan dari beberapa metode machine learning untuk memberikan rekomendasi anime yang relevan dan personal:

- 🧠 **Content-Based Filtering**  
  Sistem menganalisis konten (dalam hal ini genre) dari anime favoritmu, lalu mencocokkannya dengan anime lain yang memiliki kesamaan konten. Ini memungkinkan sistem memahami preferensimu tanpa membutuhkan data pengguna lain.
- 📊 **Term Frequency–Inverse Document Frequency (TF-IDF)**  
  Genre diubah menjadi representasi numerik menggunakan teknik TF-IDF. TF menunjukkan seberapa sering sebuah genre muncul dalam satu anime, sementara IDF memberi bobot lebih pada genre yang jarang muncul dan bobot lebih rendah pada genre yang umum. Ini membantu sistem memahami genre khas dari setiap anime.
- 👥 **K-Nearest Neighbors (KNN)**  
  Setelah semua anime diubah menjadi vektor berdasarkan genre menggunakan TF-IDF, algoritma KNN digunakan untuk mencari anime yang paling mirip berdasarkan jarak cosine antar vektor. Hasilnya adalah rekomendasi anime yang memiliki struktur genre paling dekat dengan anime pilihanmu.

Dengan kombinasi ketiga ini, sistem mampu memberikan hasil rekomendasi yang lebih sesuai dengan preferensi pengguna. 🌟

---

### ✨ Fitur Unggulan

- 🔍 **Rekomendasi Personal**: Masukkan judul anime favorit, dan dapatkan rekomendasi mirip secara otomatis.
- 📈 **Top 10 Anime Populer & Rating Tertinggi**: Berdasarkan jumlah member dan rating.
- 📂 **Eksplorasi Genre**: Lihat daftar anime dari genre tertentu.
- 🕘 **Riwayat Pencarian & Rekomendasi**: Jejak rekomendasi tetap tersedia sepanjang sesi.

---

🎯 **Siap Menemukan Anime Favoritmu Berikutnya?**
Gunakan menu navigasi di kiri untuk memulai pencarian!
""")

    st.subheader("🔥 Top 10 Anime Paling Populer")
    top_members = anime_df.sort_values(by="members", ascending=False).head(10)
    for i in range(0, len(top_members), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top_members):
                anime = top_members.iloc[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 Genre: {anime['genre']}<br>
                            ⭐ Rating: {anime['rating']}<br>
                            👥 Members: {anime['members']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.subheader("🏆 Top 10 Anime dengan Rating Tertinggi")
    top_rating = anime_df.sort_values(by="rating", ascending=False).head(10)
    for i in range(0, len(top_rating), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top_rating):
                anime = top_rating.iloc[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 Genre: {anime['genre']}<br>
                            ⭐ Rating: {anime['rating']}<br>
                            👥 Members: {anime['members']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ------------------------------
# REKOMENDASI PAGE
# ------------------------------
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime Berdasarkan Genre")

    all_titles = anime_df["name"].tolist()
    selected_title = st.selectbox("🎬 Pilih judul anime (ketik sebagian kata)", all_titles)

    if selected_title:
        anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
        anime_genre = anime_row["genre"]
        anime_rating = anime_row["rating"]

        st.markdown(f"📚 **Genre**: {anime_genre}  |  ⭐ **Rating**: {anime_rating}")

        query_vec = tfidf_vectorizer.transform([anime_genre])
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=10)

        st.success(f"🎯 Rekomendasi anime berdasarkan genre dari: {selected_title}")
        results = []
        shown = 0
        for i in indices[0]:
            row = anime_df.iloc[i]
            if row["name"] != selected_title:
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-header">{row['name']}</div>
                    <div class="anime-body">
                        📚 Genre: {row['genre']}<br>
                        ⭐ Rating: {row['rating']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                results.append({
                    "name": row["name"],
                    "genre": row["genre"],
                    "rating": row["rating"]
                })
                shown += 1
                if shown == 5:
                    break

        st.session_state.history.append(selected_title)
        st.session_state.recommendations.append({
            "query": selected_title,
            "results": results
        })

# ------------------------------
# GENRE PAGE
# ------------------------------
elif page == "📂 Genre":
    st.title("📂 Eksplorasi Anime Berdasarkan Genre")

    all_genres = sorted(set(
        g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")
    ))

    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)
    sort_by = st.radio("📊 Urutkan Berdasarkan:", ["Rating Tertinggi", "Members Terbanyak"])

    genre_filtered = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]

    if sort_by == "Rating Tertinggi":
        genre_filtered = genre_filtered.sort_values(by="rating", ascending=False)
    else:
        genre_filtered = genre_filtered.sort_values(by="members", ascending=False)

    genre_filtered = genre_filtered.head(10)

    if not genre_filtered.empty:
        st.subheader(f"📺 Top 10 Anime Genre: {selected_genre}")
        for i in range(0, len(genre_filtered), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(genre_filtered):
                    anime = genre_filtered.iloc[i + j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="anime-card">
                            <div class="anime-header">{anime['name']}</div>
                            <div class="anime-body">
                                📚 Genre: {anime['genre']}<br>
                                ⭐ Rating: {anime['rating']}<br>
                                👥 Members: {anime['members']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info(f"Belum ada anime dengan genre {selected_genre}.")
