import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Simulateur de Survie du Titanic")
st.write("Entrez les caractéristiques d'un passager pour savoir s'il aurait survécu au naufrage.")

# 2. ENTRAÎNEMENT DU MODÈLE CHAMPION (En arrière-plan)
@st.cache_resource
def train_champion_model():
    # Remarque : on remonte d'un niveau (../) pour chercher les CSV s'ils sont restés dans le dossier parent Titanic
    try:
        df_train = pd.read_csv("train.csv")
        df_test = pd.read_csv("test.csv")
    except FileNotFoundError:
        df_train = pd.read_csv("../train.csv")
        df_test = pd.read_csv("../test.csv")
        
    df_train["IsTrain"] = 1
    df_test["IsTrain"] = 0
    df_test["Survived"] = np.nan
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    # Imputations
    df_all["Age"] = df_all["Age"].fillna(df_all.groupby(["Pclass", "Sex"])["Age"].transform("median"))
    df_all["Fare"] = df_all["Fare"].fillna(df_all.groupby("Pclass")["Fare"].transform("median"))
    
    # Groupes familiaux
    df_all["LastName"] = df_all["Name"].apply(lambda x: x.split(",")[0].strip())
    df_all["FamilyID"] = df_all["LastName"] + "_" + df_all["Fare"].astype(str)
    df_all["IsWomanOrChild"] = ((df_all["Sex"] == "female") | (df_all["Age"] < 16)).astype(int)
    
    family_survival = df_all[df_all["IsTrain"] == 1].groupby("FamilyID")["Survived"].agg(["count", "mean"])
    family_survival = family_survival[family_survival["count"] > 1]
    
    df_all["Family_Survival_Rate"] = 0.5
    for idx, row in df_all.iterrows():
        f_id = row["FamilyID"]
        if f_id in family_survival.index:
            df_all.at[idx, "Family_Survival_Rate"] = family_survival.loc[f_id, "mean"]

    # Titres
    df_all["Title"] = df_all["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_all["Title"] = df_all["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare")
    df_all["Title"] = df_all["Title"].replace(["Mlle", "Ms"], "Miss")
    df_all["Title"] = df_all["Title"].replace("Mme", "Mrs")
    
    df_all = pd.get_dummies(df_all, columns=["Sex", "Title"], drop_first=True)
    df_train_final = df_all[df_all["IsTrain"] == 1].copy()
    
    features = ["Pclass", "Age", "Family_Survival_Rate", "Sex_male", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare"]
    X_train = df_train_final[features]
    y_train = df_train_final["Survived"].astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_champion_model()

# ==========================================
# 3. INTERFACE UTILISATEUR (FORMULAIRE CORRIGÉ)
# ==========================================
st.divider()
st.header("📋 Profil du passager")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sexe :", ["Femme", "Homme"])
    age = st.slider("Âge :", min_value=1, max_value=90, value=28)
    pclass = st.selectbox("Classe du billet (Pclass) :", [1, 2, 3], index=2)

with col2:
    title = st.selectbox("Titre de civilité :", ["Mr", "Miss", "Mrs", "Rare"])
    # LA MODIFICATION EST ICI : Plus naturelle, pas de spoil sur le destin de la famille !
    travel_type = st.selectbox(
        "Type de voyage :", 
        ["Voyage seul(e)", "En famille (Parents / Enfants / Époux)", "Entre amis ou en groupe (Billet partagé)"]
    )

# Traduction des entrées utilisateur pour le modèle
sex_male = 1 if sex == "Homme" else 0
title_Mr = 1 if title == "Mr" else 0
title_Miss = 1 if title == "Miss" else 0
title_Mrs = 1 if title == "Mrs" else 0
title_Rare = 1 if title == "Rare" else 0

# Conversion du type de voyage en taux de survie basé sur les statistiques historiques
if travel_type == "En famille (Parents / Enfants / Époux)":
    family_rate = 0.65  # Avantageux (entraide)
elif travel_type == "Voyage seul(e)":
    family_rate = 0.35  # Désavantageux historique
else:
    family_rate = 0.50  # Neutre (Entre amis / Groupe)

# Création du DataFrame pour la prédiction
passenger_data = pd.DataFrame([{
    "Pclass": pclass,
    "Age": age,
    "Family_Survival_Rate": family_rate,
    "Sex_male": sex_male,
    "Title_Miss": title_Miss,
    "Title_Mr": title_Mr,
    "Title_Mrs": title_Mrs,
    "Title_Rare": title_Rare
}])

# ==========================================
# 4. BOUTON DE PRÉDICTION ET AFFICHAGE
# ==========================================
st.divider()

if st.button("🔮 Lancer la prédiction", use_container_width=True):
    prediction = model.predict(passenger_data)[0]
    proba = model.predict_proba(passenger_data)[0]
    
    survival_probability = proba[1] * 100
    
    if prediction == 1:
        st.success(f"🎉 **Survivant(e) !** (Probabilité de survie : {survival_probability:.1f}%)")
        st.balloons()
    else:
        st.error(f"🌊 **Victime...** (Probabilité de survie : {survival_probability:.1f}%)")