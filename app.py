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

# Bangun model KNN + TF-IDF
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix, tfidf

knn_model, tfidf_matrix, tfidf_vectorizer = build_model(anime_df)

# CSS
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

Website ini membantu para pecinta anime menemukan tontonan baru yang cocok berdasarkan genre favorit, menggunakan teknik **TF-IDF** dan **K-Nearest Neighbors (KNN)**.
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
    st.title("🔍 Cari Rekomendasi Anime")
    anime_input = st.text_input("🎬 Masukkan judul anime favoritmu")

    if anime_input:
        anime_name = anime_input.lower().strip()
        if anime_name not in anime_df["name_lower"].values:
            st.error("Anime tidak ditemukan.")
        else:
            index = anime_df[anime_df["name_lower"] == anime_name].index[0]
            query_vec = tfidf_matrix[index]
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=6)

            original_title = anime_df.iloc[index]["name"]
            st.success(f"🎯 Rekomendasi berdasarkan: {original_title}")
            results = []
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

            st.session_state.history.append(original_title)
            st.session_state.recommendations.append({
                "query": original_title,
                "results": results
            })

# ------------------------------
# GENRE PAGE (MENGGUNAKAN KNN)
# ------------------------------
elif page == "📂 Genre":
    st.title("📂 Rekomendasi Berdasarkan Genre (via KNN)")

    all_genres = sorted(set(
        g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")
    ))
    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)

    st.subheader("🎯 5 Anime dengan Genre Paling Mirip")

    # Gunakan TF-IDF + KNN untuk cari 5 anime terdekat dengan genre ini
    genre_vec = tfidf_vectorizer.transform([selected_genre])
    distances, indices = knn_model.kneighbors(genre_vec, n_neighbors=5)

    for i in indices[0]:
        anime = anime_df.iloc[i]
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
