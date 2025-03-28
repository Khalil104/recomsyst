# 🎬 Système de Recommandation (Filtrage Collaboratif)

Ce projet implémente un **système de recommandation basé sur le filtrage collaboratif** en utilisant **Python** et **Streamlit**. Il permet aux utilisateurs de :  
✅ Ajouter manuellement des notes sur des films.  
✅ Charger un fichier CSV contenant des évaluations.  
✅ Visualiser une matrice utilisateur-film avec les notes manquantes.  
✅ Calculer la similarité entre films grâce à la **cosine similarity**.  
✅ Obtenir des recommandations de films en fonction des préférences des utilisateurs.  
✅ Effectuer une recherche pour connaître la note donnée à un film par un utilisateur spécifique et, si la note est absente, prédire la note en fonction des films similaires.

---

## 🚀 Installation et Exécution  

### 📥 1. Prérequis  
Avant de commencer, assurez-vous d'avoir :  
- **Python 3.10+** installé  
- **pip** installé pour gérer les dépendances  

### 📥 2. Installation des dépendances  
Clonez ce projet et installez les dépendances requises avec :

```bash
git clone https://github.com/ton-repo.git
cd ton-repo
pip install -r requirements.txt
