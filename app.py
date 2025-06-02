import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Konfigurasi halaman
st.set_page_config(page_title="🎥 Rekomendasi Anime", layout="wide")

# Load data dengan preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")

    # Hapus data kosong di kolom penting
    df = df.dropna(subset=["name", "genre", "rating"])

    # Hapus duplikat berdasarkan nama anime
    df = df.drop_duplicates(subset=["name"], keep="first")

    # Pastikan rating bertipe numerik
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Hapus data dengan rating < 1
    df = df[df["rating"] >= 1]

    # Reset index dan buat kolom pencarian lowercase
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
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi", "🎭 Genre"])

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "🏠 Home":
    st.title("🏠 Selamat Datang di Rekomendasi Anime")
    st.markdown("Temukan anime favoritmu berdasarkan genre yang mirip 🎯")

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

    st.subheader("🎯 Rekomendasi Baru")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations):
            # Ambil 3 rekomendasi rating tertinggi
            top3 = sorted(item["results"], key=lambda x: x["rating"], reverse=True)[:3]
            for anime in top3:
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
                st.markdown(
                    f"""
                    <div class="anime-card">
                        <div class="anime-header">{row['name']}</div>
                        <div class="anime-body">
                            📚 {row['genre']}<br>
                            ⭐ {row['rating']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
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
elif page == "🎭 Genre":
    st.title("🎭 Jelajahi Anime Berdasarkan Genre")

    # Ekstrak semua genre unik
    all_genres = set()
    for genre_list in anime_df["genre"]:
        genres = [g.strip() for g in genre_list.split(",")]
        all_genres.update(genres)
    sorted_genres = sorted(all_genres)

    selected_genre = st.selectbox("🎨 Pilih Genre", sorted_genres)

    if selected_genre:
        filtered = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]
        filtered = filtered.sort_values(by="rating", ascending=False)

        st.subheader(f"🎬 Anime dengan Genre: {selected_genre} ({len(filtered)} ditemukan)")

        for i in range(len(filtered)):
            row = filtered.iloc[i]
            st.markdown(
                f"""
                <div class="anime-card">
                    <div class="anime-header">{row['name']}</div>
                    <div class="anime-body">
                        📚 {row['genre']}<br>
                        ⭐ {row['rating']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
