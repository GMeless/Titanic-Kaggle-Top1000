# 🚢 Titanic Survival Predictor & Kaggle Top 1000

Ce projet contient le code qui m'a permis de me hisser à la **811ème place mondiale** sur le célèbre challenge Kaggle du Titanic, ainsi qu'une application web interactive pour simuler la survie des passagers en temps réel.

## 🏆 Performance Kaggle
- **Score Public :** `0.79904`
- **Classement :** 811ème mondial (Top 3% / Cour des grands 🚀)

## 🧠 Approche & Feature Engineering
Pour atteindre ce score sans tricher, j'ai développé une approche basée sur l'histoire réelle du naufrage :
1. **Imputation chirurgicale :** Remplacement des âges manquants par la valeur médiane basée sur le sexe et la classe (`Pclass`).
2. **Logique Familiale (FamilyID) :** Regroupement des passagers par Nom de famille + Prix du billet (`Fare`) pour lier leurs destins.
3. **Extraction de Titres :** Analyse textuelle des noms pour isoler les statuts sociaux (Mr, Miss, Mrs, Rare).
4. **Modèle :** Optimisation par validation croisée d'une Forêt Aléatoire (`RandomForestClassifier`).

## 💻 L'Application Web (Streamlit)
Le projet intègre une application interactive permettant de tester le destin de n'importe quel passager fictif ou historique.

### Comment lancer l'application en local :
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt