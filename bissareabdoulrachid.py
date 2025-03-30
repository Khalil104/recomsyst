import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de l'interface
st.set_page_config(page_title="Système de recommandation", layout="wide")

st.title("🎬 Système de recommandation (Filtrage collaboratif)")
st.write("Ce système recommande des films aux utilisateurs en fonction de leurs préférences passées.")

# --- Chargement des données ---
st.sidebar.header("🔹 Chargement des Données")

# Stockage des données en session (persistance)
if "ratings" not in st.session_state:
    st.session_state["ratings"] = []

# Option 1 : Chargement manuel
st.sidebar.subheader("➕ Ajouter une Note")
user_id = st.sidebar.number_input("ID de l'utilisateur :", min_value=1, step=1)
movie_id = st.sidebar.number_input("ID du film :", min_value=1, step=1)
rating = st.sidebar.slider("Note (0 à 5) :", min_value=0.0, max_value=5.0, step=0.5)

if st.sidebar.button("Ajouter la note"):
    st.session_state["ratings"].append({"user_id": user_id, "movie_id": movie_id, "rating": rating})
    st.sidebar.success("✅ Note ajoutée avec succès !")

# Option 2 : Chargement CSV
uploaded_file = st.sidebar.file_uploader("📂 Importer un fichier CSV", type=["csv"])
if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    st.session_state["ratings"] = df_csv.to_dict(orient="records")
    st.sidebar.success("✅ Fichier chargé avec succès !")

# Convertir en DataFrame
df = pd.DataFrame(st.session_state["ratings"])

# --- Vérification et gestion des doublons ---
if not df.empty:
    st.subheader("🔍 Vérification des doublons")

    duplicates = df[df.duplicated(subset=["user_id", "movie_id"], keep=False)]
    if not duplicates.empty:
        st.warning("⚠ Doublons détectés ! Les notes multiples pour un même film et utilisateur seront moyennées.")
        st.dataframe(duplicates)

    # Aggrégation des notes en prenant la moyenne pour éviter les doublons
    df = df.groupby(["user_id", "movie_id"], as_index=False).agg({"rating": "mean"})

# --- Affichage des données ---
st.subheader("📊 Données Utilisateur-Film")
if not df.empty:
    st.dataframe(df)
else:
    st.info("⚠ Aucune donnée disponible. Ajoutez des notes manuelles ou importez un CSV.")

# --- Construction de la matrice utilisateur-item ---
if not df.empty:
    pivot_table = df.pivot(index="user_id", columns="movie_id", values="rating")
    st.subheader("📌 Matrice Utilisateur-Film avec Notes Manquantes")
    st.dataframe(pivot_table.fillna("🔲"))

    # --- Calcul de la similarité item-item ---
    st.subheader("📈 Matrice de Similarité entre Films")

    # Remplacement des NaN par 0 pour le calcul de similarité
    pivot_filled = pivot_table.fillna(0)
    similarity_matrix = cosine_similarity(pivot_filled.T)

    # Conversion en DataFrame
    item_sim_df = pd.DataFrame(similarity_matrix, index=pivot_filled.columns, columns=pivot_filled.columns)
    st.dataframe(item_sim_df)

    # --- Génération du TOP N ---
    top_n = st.number_input("🎯 Sélectionner N pour Top N recommandations :", min_value=1, max_value=10, value=3)

    def get_top_n_recommendations(movie_id, n=top_n):
        if movie_id in item_sim_df.index:
            similar_items = item_sim_df[movie_id].sort_values(ascending=False)[1:n+1]
            return similar_items.index.tolist()
        return []

    # Affichage des recommandations
    selected_movie = st.selectbox("🎥 Choisissez un film pour voir les recommandations :", pivot_table.columns)
    recommended_movies = get_top_n_recommendations(selected_movie, top_n)

    if recommended_movies:
        st.success(f"✅ Films recommandés pour {selected_movie} : {recommended_movies}")
    else:
        st.warning("⚠ Aucune recommandation trouvée.")

    # --- Recherche d'un utilisateur et d'un film précis ---
    st.subheader("🔍 Rechercher une Note")
    search_user = st.number_input("ID Utilisateur à rechercher :", min_value=1, step=1)
    search_movie = st.number_input("ID Film à rechercher :", min_value=1, step=1)

    if search_user in pivot_table.index and search_movie in pivot_table.columns:
        note = pivot_table.loc[search_user, search_movie]
        if pd.isna(note):
            st.warning("⚠ Note non disponible. Prédiction en cours...")
            similar_movies = get_top_n_recommendations(search_movie, 1)
            if similar_movies:
                prediction = pivot_table[similar_movies].mean(axis=1).get(search_user, "Aucune prédiction")
                st.success(f"✅ Note estimée : {prediction}")
            else:
                st.error("❌ Impossible de prédire une note.")
        else:
            st.success(f"✅ Note réelle : {note}")
    else:
        st.info("⚠ Utilisateur ou film introuvable.")
else:
    st.warning("⚠ Aucune donnée disponible pour créer la matrice utilisateur-film.")

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; padding: 10px; font-size: 16px;">
        by <a href="https://rachidbissare.vercel.app" target="_blank" style="text-decoration: none; color: #0073b1;">
        Abdoul Rachid BISSARE</a>
    </div>
    """,
    unsafe_allow_html=True
)
