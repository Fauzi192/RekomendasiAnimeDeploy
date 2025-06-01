import streamlit as st

# ⛔ WAJIB di paling atas
st.set_page_config(page_title="🎌 Anime Recommender", layout="wide")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating"])
    df = df.reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()
    return df

anime_df = load_data()

# ------------------------------
# Build Model
# ------------------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

knn_model, tfidf_matrix = build_model(anime_df)

# ------------------------------
# CSS Styling
# ------------------------------
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

# ------------------------------
# Inisialisasi Session State
# ------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Sidebar Navigasi
# ------------------------------
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi"])

# ------------------------------
# Halaman HOME
# ------------------------------
if page == "🏠 Home":
    st.title("🏠 Selamat Datang di Anime Recommender")
    st.markdown("Temukan anime favoritmu berdasarkan genre yang mirip 🎯")

    # 🎌 Ganti animasi Jepang
    st.image("https://media.giphy.com/media/4oMoIbIQrvCjm/giphy.gif", width=300)

    st.subheader("🔥 Top 10 Anime Paling Populer")
    top10 = anime_df.sort_values(by="rating", ascending=False).head(10)

    for i in range(0, len(top10), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top10):
                anime = top10.iloc[i + j]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div class="anime-card">
                            <div class="anime-header">{anime['name']}</div>
                            <div class="anime-body">
                                📚 Genre: {anime['genre']}<br>
                                ⭐ Rating: {anime['rating']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.subheader("🕘 Riwayat Pencarian")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(f"🔎 {item}")
    else:
        st.info("Belum ada pencarian yang dilakukan.")

    st.subheader("🎯 Rekomendasi Sebelumnya")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations):
            st.markdown(f"<h5 style='color:#f04e7c;'>📌 Untuk: <i>{item['query']}</i></h5>", unsafe_allow_html=True)
            for anime in item["results"]:
                st.markdown(
                    f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 {anime['genre']}<br>
                            ⭐ {anime['rating']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("Belum ada hasil rekomendasi.")

# ------------------------------
# Halaman REKOMENDASI
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
            for i in indices[0][1:]:
                row = anime_df.iloc[i]
                results.append({
                    "name": row["name"],
                    "genre": row["genre"],
                    "rating": row["rating"]
                })

            # Simpan hasil dan history (tidak ditampilkan langsung)
            st.session_state.history.append(original_title)
            st.session_state.recommendations.append({
                "query": original_title,
                "results": results
            })

            st.success("🎯 Rekomendasi telah ditambahkan ke halaman Home.")
