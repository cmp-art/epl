# ⚽ EPL Match Prediction API: End-to-End Machine Learning Pipeline

![CI/CD Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker)

## 📌 System Overview
This project is a fully containerized, production-ready machine learning API that predicts English Premier League match outcomes. Moving beyond static Jupyter notebooks, this system implements automated data ingestion, real-time model inference using XGBoost, SHAP-based explainability, and persistent prediction logging via PostgreSQL. 

The architecture is designed with MLOps best practices, featuring automated testing and CI/CD pipelines to ensure reliability across deployments.

## 🏗️ System Architecture & Process Automation
* **Data Ingestion:** Automated weekly scripts fetch live match data via external APIs to continuously update rolling ELO ratings and team form.
* **Inference Engine:** XGBoost model exposed via a low-latency **FastAPI** endpoint.
* **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** transforms the "black box" prediction into transparent, feature-level impact scores.
* **Database Logging:** Production predictions are securely logged to a **Neon serverless PostgreSQL** database via `psycopg2` for performance monitoring.
* **DevOps & CI/CD:** Packaged in a **Docker** container. **GitHub Actions** automates the testing environment (`pytest`) on every push to the `main` branch.

## 🚀 API Usage
The API accepts match parameters and returns the predicted outcome alongside the top three mathematical features driving the decision.

**Endpoint:** `/predict` (POST)

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "hometeam": "Arsenal",
  "awayteam": "Chelsea"
}'