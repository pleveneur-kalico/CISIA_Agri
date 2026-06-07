# Dockerfile — API de détection d'anomalies parcellaires (CISIA Agriculture)
# Certification CISIA — Compétence C7 (Architecture cible)

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="CISIA Agriculture"
LABEL description="API REST de détection d'anomalies parcellaires par XGBoost"

# Répertoire de travail
WORKDIR /app

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et des données
COPY api/ ./api/
COPY data/modele_xgboost_optuna.pkl ./data/
COPY data/preprocessor.pkl ./data/
COPY data/scaler.pkl ./data/

# Exposition du port API
EXPOSE 8000

# Santé du conteneur
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Lancement de l'API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
