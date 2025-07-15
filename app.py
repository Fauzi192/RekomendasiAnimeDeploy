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

# Session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# CSS kustom
st.markdown("""
<style>
    body, .main {
        background-color: #fefefe !important;
        color: #222222;
    }
    .stApp {
        background-color: #fefefe;
    }
    .anime-card {
        background-color: #fffdfd;
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
- 📊 **TF-IDF** (representasi vektor genre)
- 👥 **KNN** (menghitung kemiripan antar anime)

---

### ✨ Fitur Unggulan

- 🔍 Rekomendasi berdasarkan input judul anime
- 📈 Top 10 anime populer dan dengan rating tertinggi
- 📂 Eksplorasi berdasarkan genre
- 🕘 Riwayat pencarian dan hasil rekomendasi tersimpan selama sesi

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

    st.subheader("🕘 Riwayat Pencarian")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"🔎 {item}")
    else:
        st.info("Belum ada pencarian.")

    st.subheader("🎯 Rekomendasi Terakhir")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations[-5:]):
            st.markdown(f"**📌 Dari**: {item['query']}")
            for anime in item['results']:
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
        st.info("Belum ada rekomendasi.")

# ------------------------------
# REKOMENDASI PAGE
# ------------------------------
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime")

    input_text = st.text_input("🎬 Masukkan sebagian judul anime")
    type_options = ["Semua"] + sorted(anime_df["type"].dropna().unique())
    selected_type = st.selectbox("🎞️ Pilih Type Anime", type_options)

    if input_text:
        matches = anime_df[anime_df["name_lower"].str.contains(input_text.lower())]
        if not matches.empty:
            selected_title = st.selectbox("🔽 Pilih judul", matches["name"].unique())
            anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
            anime_genre = anime_row["genre"]

            st.markdown(f"📚 **Genre**: {anime_genre}  |  ⭐ **Rating**: {anime_row['rating']}")

            query_vec = tfidf_vectorizer.transform([anime_genre])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=20)

            st.success(f"🎯 Rekomendasi berdasarkan genre dari: {selected_title}")
            results = []
            shown = 0
            for i in indices[0]:
                result = anime_df.iloc[i]
                if result["name"] != selected_title:
                    if selected_type == "Semua" or result["type"] == selected_type:
                        st.markdown(f"""
                        <div class="anime-card">
                            <div class="anime-header">{result['name']}</div>
                            <div class="anime-body">
                                📚 Genre: {result['genre']}<br>
                                ⭐ Rating: {result['rating']}<br>
                                🎞️ Type: {result['type']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        results.append({
                            "name": result["name"],
                            "genre": result["genre"],
                            "rating": result["rating"],
                            "type": result["type"]
                        })
                        shown += 1
                        if shown == 5:
                            break

            st.session_state.history.append(f"{selected_title} (Type: {selected_type})")
            st.session_state.recommendations.append({
                "query": f"{selected_title} (Type: {selected_type})",
                "results": results
            })
        else:
            st.warning("Judul tidak ditemukan.")

# ------------------------------
# GENRE PAGE
# ------------------------------
elif page == "📂 Genre":
    st.title("📂 Eksplorasi Rekomendasi Berdasarkan Genre (KNN)")

    all_genres = sorted(set(
        g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")
    ))

    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)
    sort_by = st.radio("📊 Urutkan Hasil Berdasarkan:", ["Rating", "Members"])

    if selected_genre:
        # Cari anime dengan genre tersebut
        matching_anime = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]
        
        if matching_anime.empty:
            st.warning("Tidak ditemukan anime dengan genre tersebut.")
        else:
            # Ambil genre dari anime pertama
            genre_text = matching_anime.iloc[0]["genre"]
            query_vec = tfidf_vectorizer.transform([genre_text])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=30)

            # Ambil anime hasil KNN yang memiliki genre yang cocok
            knn_results = []
            for i in indices[0]:
                anime = anime_df.iloc[i]
                if selected_genre.lower() in anime["genre"].lower():
                    knn_results.append({
                        "name": anime["name"],
                        "genre": anime["genre"],
                        "rating": anime["rating"],
                        "members": anime["members"]
                    })

            # Sort hasil
            if sort_by == "Rating":
                knn_results = sorted(knn_results, key=lambda x: x["rating"], reverse=True)
            else:
                knn_results = sorted(knn_results, key=lambda x: x["members"], reverse=True)

            knn_results = knn_results[:5]  # Ambil 5 teratas

            if knn_results:
                st.subheader(f"🎯 Rekomendasi Anime Genre '{selected_genre}' Berdasarkan KNN")
                for anime in knn_results:
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

                # Simpan ke history
                st.session_state.history.append(f"Genre: {selected_genre}")
                st.session_state.recommendations.append({
                    "query": f"Genre: {selected_genre}",
                    "results": knn_results
                })
            else:
                st.info("Belum ada hasil rekomendasi yang cocok dengan genre tersebut.")
