"""
API REST de détection d'anomalies parcellaires — CISIA Agriculture
FastAPI + XGBoost + SHAP
"""

import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELE_PATH = DATA_DIR / "modele_xgboost_optuna.pkl"
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"

# ─── Chargement du modèle et du préprocesseur ────────────────────────────────
app = FastAPI(
    title="API Détection d'Anomalies Parcellaires",
    description="Système de détection d'anomalies agricoles par Machine Learning — Certification CISIA",
    version="1.0.0"
)

# CORS — autorise les appels depuis la présentation HTML ouverte en local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement au démarrage
try:
    modele = joblib.load(MODELE_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    # SHAP TreeExplainer (optimisé pour XGBoost, sans background dataset)
    explainer = shap.TreeExplainer(modele)
    print(f"Modèle chargé : {MODELE_PATH}")
    print(f"Préprocesseur chargé : {PREPROCESSOR_PATH}")
    print(f"Scaler chargé : {SCALER_PATH}")
    print(f"SHAP Explainer initialisé (TreeExplainer)")
except FileNotFoundError as e:
    print(f"Fichier(s) manquant(s) : {e}")
    print("Exécutez d'abord le notebook 01_application.ipynb pour générer les .pkl")
    modele = None
    preprocessor = None
    scaler = None
    explainer = None

# ─── Schémas Pydantic ─────────────────────────────────────────────────────────
class ObservationInput(BaseModel):
    """Caractéristiques d'une observation à évaluer."""
    temperature: float = Field(..., ge=-20, le=45, description="Température ambiante (°C)")
    humidite: float = Field(..., ge=0, le=100, description="Humidité relative (%)")
    pluviometrie_mm: float = Field(..., ge=0, le=300, description="Précipitations 24h (mm)")
    ndvi: float = Field(..., ge=-1, le=1, description="Indice de végétation NDVI")
    capteur: str = Field(..., description="Type de capteur (Satellite, Drone, Station_Sol, Manuel)")
    stade_culture: str = Field(..., description="Stade phénologique (Semis, Levée, Tallage, Floraison, Maturation, Récolte)")
    rendement_estime: float = Field(..., ge=0, description="Rendement estimé (t/ha)")
    rendement_moyen_zone: float = Field(..., ge=0, description="Rendement moyen historique zone (t/ha)")
    type_culture: str = Field(..., description="Type de culture (Blé, Maïs, Vigne, Tournesol, Colza)")
    type_sol: str = Field(..., description="Type de sol (Argileux, Limoneux, Sableux, Calcaire)")
    region: str = Field(..., description="Région administrative")
    irrigation: str = Field(..., description="Système d'irrigation (Goutte-à-goutte, Aspersion, Gravitaire, Aucune)")
    surface_ha: float = Field(..., gt=0, description="Surface de la parcelle (ha)")
    age_culture_jours: int = Field(..., ge=0, description="Âge de la culture (jours depuis mise en culture)")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 22.5,
                "humidite": 65.0,
                "pluviometrie_mm": 3.2,
                "ndvi": 0.68,
                "capteur": "Drone",
                "stade_culture": "Floraison",
                "rendement_estime": 7.8,
                "rendement_moyen_zone": 8.1,
                "type_culture": "Blé",
                "type_sol": "Limoneux",
                "region": "Occitanie",
                "irrigation": "Goutte-à-goutte",
                "surface_ha": 12.5,
                "age_culture_jours": 95
            }
        }

class ContributionSHAP(BaseModel):
    """Contribution SHAP d'une variable pour une prédiction."""
    variable: str = Field(..., description="Nom de la variable")
    contribution: float = Field(..., description="Valeur SHAP (positive = augmente le risque, négative = le réduit)")
    direction: str = Field(..., description="Sens de l'impact : 'augmente le risque' ou 'réduit le risque'")

class PredictionOutput(BaseModel):
    """Résultat de la prédiction."""
    anomalie_predite: int = Field(..., description="0 = Normal, 1 = Anomalie")
    probabilite_anomalie: float = Field(..., description="Probabilité d'anomalie (0 à 1)")
    facteurs_principaux: List[str] = Field(default=[], description="Facteurs les plus influents")
    contributions_shap: List[ContributionSHAP] = Field(default=[], description="Détail SHAP des 5 variables les plus influentes")
    message: str = Field(..., description="Interprétation humaine")

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Vérification de l'état du service."""
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "ok",
        "modele": "XGBoost",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(obs: ObservationInput):
    """Prédit si une observation correspond à une anomalie parcellaire."""
    if modele is None or preprocessor is None or scaler is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord le notebook d'entraînement."
        )

    # Construction du DataFrame avec les colonnes dans l'ordre attendu
    df_input = pd.DataFrame([{
        'Temperature': obs.temperature,
        'Humidite': obs.humidite,
        'Pluviometrie_mm': obs.pluviometrie_mm,
        'NDVI': obs.ndvi,
        'Capteur': obs.capteur,
        'StadeCulture': obs.stade_culture,
        'RendementEstime_t_ha': obs.rendement_estime,
        'RendementMoyenZone_t_ha': obs.rendement_moyen_zone,
        'Region': obs.region,
        'TypeCulture': obs.type_culture,
        'TypeSol': obs.type_sol,
        'Irrigation': obs.irrigation,
        'Surface_ha': obs.surface_ha,
        'AgeCulture_jours': obs.age_culture_jours,
    }])

    # Feature engineering (identique à la Phase 3 du notebook)
    df_input['EcartRendement_t_ha'] = (
        df_input['RendementEstime_t_ha'] - df_input['RendementMoyenZone_t_ha']
    )
    df_input['RatioRendement'] = np.where(
        df_input['RendementMoyenZone_t_ha'] > 0,
        df_input['RendementEstime_t_ha'] / df_input['RendementMoyenZone_t_ha'],
        1.0
    )

    # Application du préprocesseur (OneHot + Ordinal + passthrough)
    X_encoded = preprocessor.transform(df_input)

    # Standardisation UNIQUEMENT des colonnes numériques (passthrough)
    noms_features = preprocessor.get_feature_names_out()
    idx_debut_num = sum(1 for n in noms_features if n.startswith('onehot__') or n.startswith('ordinal__'))
    X_encoded[:, idx_debut_num:] = scaler.transform(X_encoded[:, idx_debut_num:])

    # Prédiction
    proba = float(modele.predict_proba(X_encoded)[0, 1])
    prediction = int(proba >= 0.5)

    # Calcul des valeurs SHAP pour cette observation
    shap_values = explainer.shap_values(X_encoded)
    # Pour XGBoost binaire, shap_values est une matrice (1, n_features)
    shap_row = shap_values[0]

    # Associer chaque feature à sa valeur SHAP et son nom
    feature_contributions = []
    for i, nom in enumerate(noms_features):
        # Simplifier les noms : enlever les préfixes onehot__ / ordinal__ / remainder__
        nom_simple = nom
        for prefix in ['onehot__', 'ordinal__', 'remainder__']:
            if nom_simple.startswith(prefix):
                nom_simple = nom_simple[len(prefix):]
        feature_contributions.append({
            'variable': nom_simple,
            'valeur_shap': float(shap_row[i]),
            'valeur_absolue': abs(float(shap_row[i]))
        })

    # Trier par impact absolu décroissant et prendre les 5 plus influents
    feature_contributions.sort(key=lambda x: x['valeur_absolue'], reverse=True)
    top5 = feature_contributions[:5]

    # Construire les contributions SHAP détaillées
    contributions_shap = []
    facteurs_principaux = []
    for f in top5:
        direction = "augmente le risque" if f['valeur_shap'] > 0 else "réduit le risque"
        contributions_shap.append(ContributionSHAP(
            variable=f['variable'],
            contribution=round(f['valeur_shap'], 5),
            direction=direction
        ))
        # Pour la liste courte, on prend les 3 premiers
        if len(facteurs_principaux) < 3:
            facteurs_principaux.append(f['variable'])

    # Message interprétable
    if prediction == 1:
        if proba > 0.8:
            message = "Anomalie détectée avec une confiance élevée — inspection recommandée."
        else:
            message = "Anomalie probable — vérification conseillée."
    else:
        if proba < 0.2:
            message = "Situation normale avec une confiance élevée."
        else:
            message = "Situation probablement normale — surveillance de routine."

    return PredictionOutput(
        anomalie_predite=prediction,
        probabilite_anomalie=round(proba, 4),
        facteurs_principaux=facteurs_principaux,
        contributions_shap=contributions_shap,
        message=message
    )

# ─── Lancement ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
