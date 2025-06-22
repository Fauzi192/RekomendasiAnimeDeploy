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
    df = df.dropna(subset=["name", "genre", "rating", "members", "type"])
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

# Session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

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

# Sidebar navigasi
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi", "📂 Genre", "🎞️ Type"])

# Halaman Home
if page == "🏠 Home":
    st.title("🎌 Rekomendasi Anime Favorit")
    st.markdown("""
Selamat datang di website **Rekomendasi Anime Favorit**! 🎉

Website ini dirancang untuk membantu kamu menemukan anime yang cocok dengan selera pribadi hanya dengan beberapa klik saja. Dengan pertumbuhan industri anime yang sangat pesat, ratusan bahkan ribuan judul baru bermunculan setiap tahun. Hal ini dapat menyulitkan para penggemar — terutama yang baru mulai menyukai anime — untuk memilih tontonan yang sesuai dengan preferensi mereka.

Melalui website ini, kamu tidak hanya bisa mendapatkan saran judul anime berdasarkan anime yang kamu sukai, tetapi juga bisa **menjelajahi rekomendasi berdasarkan genre dan jenis tayangan (type)** seperti TV Series, Movie, OVA, dan lainnya.

---

### ⚙️ Teknologi Cerdas di Balik Layar

Sistem ini dibangun dengan pendekatan modern menggunakan **machine learning** dan pengolahan data cerdas. Berikut komponen utamanya:

- 🧠 **Content-Based Filtering**  
  Sistem merekomendasikan anime baru dengan menganalisis kesamaan konten (*genre*) dari anime yang sudah kamu sukai.

- 📊 **TF-IDF (Term Frequency–Inverse Document Frequency)**  
  Setiap genre diubah menjadi angka atau vektor berbobot untuk menunjukkan tingkat kepentingan suatu genre dalam kumpulan data.

- 👥 **K-Nearest Neighbors (KNN)**  
  Menghitung kemiripan antar anime berdasarkan vektor genre, lalu memilih anime yang paling dekat (mirip) untuk direkomendasikan.

Teknologi ini memastikan bahwa rekomendasi yang kamu terima **berdasarkan kesamaan yang terukur**, bukan sekadar daftar acak.

---

### 🔍 Fitur Utama Website

- 🎬 **Rekomendasi Personal Berdasarkan Judul**  
  Masukkan judul anime favoritmu, lalu sistem akan menampilkan rekomendasi anime lain yang memiliki genre serupa.

- 📂 **Eksplorasi Berdasarkan Genre**  
  Kamu dapat memilih genre tertentu seperti "Action", "Romance", "Comedy", dll., dan sistem akan memberikan rekomendasi terbaik.

- 🎞️ **Eksplorasi Berdasarkan Type (Format Tayang)**  
  Temukan anime dalam format tayang seperti TV Series, Movie, OVA, atau Web.

- 📈 **Top Anime**  
  Lihat daftar anime paling populer dan dengan rating tertinggi dalam database.

- 🕘 **Riwayat Interaksi Pengguna**  
  Setiap judul yang kamu cari atau genre/type yang kamu pilih akan tercatat dan dapat dilihat kembali.

---

### 🤖 Teknologi yang Digunakan

- **Python & Pandas** — untuk pemrosesan dan manajemen data.
- **Scikit-learn** — untuk membangun model machine learning.
- **TF-IDF Vectorizer** — untuk mengubah teks genre menjadi vektor numerik.
- **K-Nearest Neighbors** — sebagai algoritma utama pencari kemiripan.
- **Streamlit** — untuk membangun antarmuka interaktif berbasis web dengan cepat dan responsif.

---

🎯 **Siap Menemukan Anime Terbaik Sesuai Selera Kamu?**

Gunakan menu di samping kiri untuk menjelajah semua fitur yang tersedia.
""")


    st.subheader("🔥 Top 10 Anime Paling Populer")
    top_members = anime_df.sort_values(by="members", ascending=False).head(10)
    for _, anime in top_members.iterrows():
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

# Halaman Rekomendasi
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime")
    input_text = st.text_input("🎬 Masukkan sebagian judul anime")

    if input_text:
        matches = anime_df[anime_df["name_lower"].str.contains(input_text.lower())]
        if not matches.empty:
            selected_title = st.selectbox("🔽 Pilih judul", matches["name"].unique())
            anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
            query_vec = tfidf_vectorizer.transform([anime_row["genre"]])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=10)

            st.success(f"🎯 Rekomendasi berdasarkan genre dari: {selected_title}")
            for i in indices[0]:
                result = anime_df.iloc[i]
                if result["name"] != selected_title:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{result['name']}</div>
                        <div class="anime-body">
                            📚 Genre: {result['genre']}<br>
                            ⭐ Rating: {result['rating']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Halaman Genre
elif page == "📂 Genre":
    st.title("📂 Eksplorasi Berdasarkan Genre")
    all_genres = sorted(set(g.strip() for genres in anime_df["genre"] for g in genres.split(",")))
    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)

    if selected_genre:
        sample = anime_df[anime_df["genre"].str.contains(selected_genre)].iloc[0]
        query_vec = tfidf_vectorizer.transform([sample["genre"]])
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=30)

        results = []
        for i in indices[0]:
            anime = anime_df.iloc[i]
            if selected_genre.lower() in anime["genre"].lower():
                results.append(anime)
        
        for _, anime in pd.DataFrame(results).sort_values(by="rating", ascending=False).head(5).iterrows():
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

# Halaman Type
elif page == "🎞️ Type":
    st.title("🎞️ Eksplorasi Berdasarkan Type")
    all_types = anime_df["type"].unique()
    selected_type = st.selectbox("🎬 Pilih Type", sorted(all_types))

    if selected_type:
        type_df = anime_df[anime_df["type"] == selected_type]
        sample = type_df.iloc[0]
        query_vec = tfidf_vectorizer.transform([sample["genre"]])
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=30)

        results = []
        for i in indices[0]:
            anime = anime_df.iloc[i]
            if anime["type"] == selected_type:
                results.append(anime)

        for _, anime in pd.DataFrame(results).sort_values(by="rating", ascending=False).head(5).iterrows():
            st.markdown(f"""
            <div class="anime-card">
                <div class="anime-header">{anime['name']}</div>
                <div class="anime-body">
                    📚 Genre: {anime['genre']}<br>
                    ⭐ Rating: {anime['rating']}<br>
                    👥 Members: {anime['members']}<br>
                    🎞️ Type: {anime['type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
