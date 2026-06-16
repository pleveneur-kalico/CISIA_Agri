# CISIA Agriculture — Détection d'Anomalies Parcellaires

Système de détection d'anomalies parcellaires par Machine Learning pour une coopérative agricole régionale.

## Contexte

L'agriculture de précision exploite des données issues de capteurs au sol, satellites, drones et stations météo pour détecter précocement les anomalies sur les parcelles agricoles (stress hydrique, maladies, dysfonctionnements d'irrigation).

Ce projet vise à concevoir un modèle de classification binaire capable de distinguer les situations **normales** des **anomalies nécessitant une intervention**, à partir de données d'observation historiques.

## Structure du projet

```
CISIA_Agri/
README.md # Ce fichier décrit la structure du projet, et comment le mettre en oeuvre et tester
requirements.txt # Liste des bibliothèques python à installer
.gitignore # Fichier du dossier ignoré pour un envoi du projet sur github
Dockerfile # Conteneurisation de l'API vi Docker
presentation.html # Présentation Reveal.js (client final)
journal_bord.md # Journal de bord réalisé sur le développement effectué
api/
main.py # API FastAPI
data/
parcelles.csv # 500 parcelles agricoles
observations.csv # 10 000 relevés d'observation
notebooks/
01_application.ipynb # Notebook principal (application)
02_reeentrainement.ipynb # MLOps : réentraînement & comparaison
venv/ # Environnement virtuel Python
```

## Installation

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Notebooks

Deux notebooks Jupyter sont fournis, à exécuter **dans l'ordre**.

### Lancer Jupyter

```bash
jupyter notebook
```

Dans l'interface Jupyter, naviguer vers le dossier `notebooks/` et ouvrir les fichiers `.ipynb`.

### 01_application.ipynb — Pipeline principal

**Objectif :** Construire, entraîner et évaluer le modèle de détection d'anomalies.

| Phase | Contenu |
|-------|---------|
| 1 | Ingestion des données, fusion, mise en conformité RGPD |
| 2 | Nettoyage (valeurs aberrantes, imputation), analyse exploratoire |
| 3 | Feature engineering (âge culture, écarts rendement), encodage One-Hot et Ordinal |
| 4 | Modélisation : Régression Logistique → Random Forest → XGBoost, sélection automatique du meilleur + Optuna + MLflow |
| 5 | Évaluation (ROC, matrice de confusion, SHAP), architecture cible |

**À exécuter en premier.** Il génère les fichiers nécessaires au reste du projet :
- `data/modele_final_optuna.pkl` — modèle entraîné (type détecté automatiquement : Random Forest ou XGBoost)
- `data/preprocessor.pkl` — préprocesseur (OneHot + Ordinal)
- `data/scaler.pkl` — standardisation

### 02_reeentrainement.ipynb — MLOps : amélioration continue

**Objectif :** Simuler un cycle complet de réentraînement (compétence C9).

| Étape | Contenu |
|-------|---------|
| 1 | Modèle V1 entraîné sur les 50 % d'observations les plus anciennes |
| 2 | Modèle V2 réentraîné sur 100 % des données (nouvelles données simulées) |
| 3 | Comparaison V1 vs V2 : ROC-AUC, Recall, F1, matrices de confusion |
| 4 | PSI (Population Stability Index) — détection de dérive des distributions |
| 5 | Décision documentée : quel modèle déployer en production ? |

**À exécuter après le notebook 01** (il utilise le même pipeline).

---

## API de prédiction

Une API REST permet d'interroger le modèle entraîné en conditions réelles.

### Lancement

```bash
# Depuis la racine du projet
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

L'API est alors accessible sur `http://localhost:8000`.

### Endpoints

| Méthode | Chemin | Description |
|---------|--------|-------------|
| `GET` | `/health` | Vérification de l'état du service |
| `GET` | `/docs` | Documentation Swagger interactive |
| `POST` | `/predict` | Prédiction d'anomalie pour une observation |

### Test avec Swagger

Ouvrir `http://localhost:8000/docs` dans un navigateur, cliquer sur `POST /predict` → "Try it out", puis utiliser l'exemple fourni.

### Test avec curl (PowerShell)

```powershell
$body = @{
temperature = 22.5
humidite = 65.0
pluviometrie_mm = 3.2
ndvi = 0.68
capteur = "Drone"
stade_culture = "Floraison"
rendement_estime = 7.8
rendement_moyen_zone = 8.1
type_culture = "Blé"
type_sol = "Limoneux"
region = "Occitanie"
irrigation = "Goutte-à-goutte"
surface_ha = 12.5
age_culture_jours = 95
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json
```

### Réponse attendue

```json
{
  "anomalie_predite": 0,
  "probabilite_anomalie": 0.02,
  "facteurs_principaux": ["NDVI", "RatioRendement", "Pluviometrie_mm"],
  "contributions_shap": [
    {
      "variable": "NDVI",
      "contribution": -0.09,
      "direction": "réduit le risque"
    },
    {
      "variable": "RatioRendement",
      "contribution": -0.08,
      "direction": "réduit le risque"
    },
    {
      "variable": "Pluviometrie_mm",
      "contribution": -0.05,
      "direction": "réduit le risque"
    },
    {
      "variable": "RendementEstime_t_ha",
      "contribution": -0.04,
      "direction": "réduit le risque"
    },
    {
      "variable": "Humidite",
      "contribution": -0.03,
      "direction": "réduit le risque"
    }
  ],
  "message": "Situation normale avec une confiance élevée."
}
```

### Docker

```bash
# Le logiciel Docker doit être lancé

# Construction de l'image
docker build -t cisia-agri .

# Lancement du conteneur
# Depuis le terminal, lancer :
docker run --name CISIA_Agri -p 8000:8000 cisia-agri
```

L'option `--name` évite le nom aléatoire attribué par Docker (ex: `gallant_bohr`).
Une fois lancé, on peut tester l'api via Swagger :

Pour relancer un conteneur déjà créé :
```bash
docker start CISIA_Agri
```

Pour arrêter le conteneur :
```bash
docker stop CISIA_Agri
```

Pour supprimer le conteneur (avant de le recréer) :
```bash
docker rm -f CISIA_Agri
```

## Présentation Client

Une présentation **Reveal.js** est disponible dans le fichier `presentation.html`.

**Contenu :** 28 slides couvrant l'ensemble de la démarche projet, de la problématique à l'amélioration continue, sans aucun code — destinée à un public métier (coopérative agricole).

**Lancement :**
- **Via Docker** : accéder à `http://localhost:8000/` après avoir lancé le conteneur (le test API intégré fonctionne automatiquement)
- **En local** : ouvrir simplement `presentation.html` dans un navigateur web (double-clic) — l'API doit être lancée séparément via `uvicorn api.main:app --reload`

Navigation :
- Flèches droite/gauche pour avancer/reculer
- `ESPACE` pour avancer
- `Échap` pour la vue d'ensemble
- `F` pour le mode plein écran


## Stack technique

| Catégorie | Outils |
|---|---|
| Manipulation de données | pandas, numpy |
| Visualisation | matplotlib, seaborn, missingno |
| Machine Learning | scikit-learn, XGBoost (sélection automatique RF / XGBoost) |
| Gestion déséquilibre | imbalanced-learn (SMOTE) |
| Optimisation | Optuna |
| Explicabilité | SHAP |
| MLOps | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Conteneurisation | Docker |

## Objectif

Classification binaire supervisée — variable cible : `AnomalieLabel` (0 = normal, 1 = anomalie).

## Données

- **parcelles.csv** : 500 parcelles (identifiant, région, surface, type de culture, type de sol, irrigation)
- **observations.csv** : 10 000 relevés (température, humidité, pluviométrie, NDVI, stade culture, rendement)

Les deux fichiers sont liés par la colonne `ParcelleID`.

## Licence

Projet réalisé dans le cadre de la certification CISIA.
