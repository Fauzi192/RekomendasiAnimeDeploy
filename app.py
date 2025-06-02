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
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

knn_model, tfidf_matrix = build_model(anime_df)

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

# Navigasi sidebar
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi", "📂 Berdasarkan Genre"])

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "🏠 Home":
    st.title("🏠 Selamat Datang di Rekomendasi Anime")
    st.markdown("""
    Selamat datang di website **Rekomendasi Anime Berbasis Genre**! 🎉

    Website ini dirancang khusus bagi para penggemar anime yang ingin menemukan judul-judul anime baru dan menarik sesuai dengan selera mereka. Dengan menggunakan metode **Content-Based Filtering** dan algoritma **K-Nearest Neighbor (KNN)**, sistem kami mampu memberikan rekomendasi anime yang memiliki genre serupa dengan anime favorit Anda.

    Apa yang membuat sistem kami unik?
    - **Personalisasi Rekomendasi**: Rekomendasi disesuaikan berdasarkan genre anime yang Anda pilih atau masukkan, sehingga hasilnya lebih relevan dan sesuai dengan preferensi pribadi.
    - **Data Lengkap dan Terpercaya**: Kami menggunakan data anime yang mencakup genre, rating, dan jumlah anggota (members) yang besar sebagai indikator popularitas, sehingga rekomendasi kami bukan hanya relevan tapi juga berkualitas.
    - **Tampilan Interaktif**: Antarmuka yang mudah digunakan dengan navigasi yang jelas, sehingga Anda dapat mencari anime, melihat anime populer, dan menjelajahi berdasarkan genre favorit dengan nyaman.
    - **Riwayat dan Rekomendasi Terbaru**: Anda bisa melihat kembali riwayat pencarian dan hasil rekomendasi terbaru yang sudah Anda lakukan selama sesi berjalan.
    
    Bagaimana cara menggunakan website ini?
    - Pada menu **Rekomendasi**, Anda cukup memasukkan judul anime favorit Anda, dan sistem akan mencari anime lain yang memiliki genre mirip.
    - Pada menu **Berdasarkan Genre**, Anda dapat memilih genre dari dropdown untuk menampilkan daftar anime yang sesuai genre tersebut.
    - Di halaman utama, Anda juga dapat melihat **Top 10 Anime Paling Populer** berdasarkan jumlah anggota, serta **Top 10 Anime dengan Rating Tertinggi**.

    Dengan sistem ini, kami berharap membantu Anda menemukan anime-anime baru yang seru dan sesuai dengan selera Anda tanpa harus bingung memilih dari banyaknya judul yang tersedia.

    Selamat menjelajahi dunia anime dan temukan rekomendasi terbaik hanya di sini! 🌟
    """)

    st.subheader("🔥 Top 10 Anime Paling Populer (berdasarkan jumlah members)")
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

    st.subheader("🕘 Riwayat Pencarian")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(f"🔎 {item}")
    else:
        st.info("Belum ada pencarian yang dilakukan.")

    st.subheader("🎯 Rekomendasi Baru")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations):
            top3 = sorted(item["results"], key=lambda x: x["rating"], reverse=True)[:3]
            for anime in top3:
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-header">{anime['name']}</div>
                    <div class="anime-body">
                        📚 {anime['genre']}<br>
                        ⭐ {anime['rating']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Belum ada hasil rekomendasi.")

# ------------------------------
# REKOMENDASI PAGE
# ------------------------------
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime")
    st.markdown("Masukkan nama anime favoritmu dan dapatkan rekomendasi genre sejenis 🎌")

    anime_name_input = st.text_input("🎬 Masukkan judul anime")

    if anime_name_input:
        anime_name = anime_name_input.strip().lower()

        if anime_name not in anime_df["name_lower"].values:
            st.error("Anime tidak ditemukan. Pastikan penulisan judul sudah benar.")
        else:
            index = anime_df[anime_df["name_lower"] == anime_name].index[0]
            query_vec = tfidf_matrix[index]
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=6)

            original_title = anime_df.iloc[index]["name"]
            results = []
            st.success(f"🎯 Rekomendasi berdasarkan: {original_title}")
            for i in indices[0][1:]:
                row = anime_df.iloc[i]
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-header">{row['name']}</div>
                    <div class="anime-body">
                        📚 {row['genre']}<br>
                        ⭐ {row['rating']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                results.append({
                    "name": row["name"],
                    "genre": row["genre"],
                    "rating": row["rating"]
                })

            # Simpan ke history & rekomendasi
            st.session_state.history.append(original_title)
            st.session_state.recommendations.append({
                "query": original_title,
                "results": results
            })

# ------------------------------
# GENRE PAGE
# ------------------------------
elif page == "📂 Berdasarkan Genre":
    st.title("📂 Eksplorasi Anime Berdasarkan Genre")

    all_genres = sorted(set(
        g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")
    ))

    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)

    genre_filtered = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]

    if not genre_filtered.empty:
        st.subheader(f"📺 Anime dengan Genre: {selected_genre}")
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

