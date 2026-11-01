# ⚽ EPL Match Prediction API: End-to-End Machine Learning Pipeline

## 📌 System Overview

This project is a fully containerized, production-ready machine learning API that predicts English Premier League match outcomes. Moving beyond static Jupyter notebooks, this system implements automated data ingestion, real-time model inference using XGBoost, SHAP-based explainability, and persistent prediction logging via PostgreSQL.

The architecture is designed with MLOps best practices, featuring automated testing and CI/CD pipelines to ensure reliability across deployments.

## 🏗️ System Architecture & Process Automation

* **Data Ingestion:** Automated weekly scripts fetch live match data via external APIs to continuously update rolling ELO ratings and team form.

* **Inference Engine:** XGBoost model exposed via a low-latency **FastAPI** endpoint.

* **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** transforms the "black box" prediction into transparent, feature-level impact scores.

* **Database Logging:** Production predictions are securely logged to a **Neon serverless PostgreSQL** database via `psycopg2` for performance monitoring.

* **DevOps & CI/CD:** Packaged in a **Docker** container. **GitHub Actions** automates the testing environment (`pytest`) on every push to the `main` branch.

## 💻 Local Installation & Setup

To run this pipeline locally, you will need Python 3.9+ installed on your machine.

**1. Clone the repository and create a virtual environment:**

```bash
git clone https://github.com/yourusername/epl-prediction-api.git
cd epl-prediction-api
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory based on the provided `.env.example`.

```bash
# .env
FOOTBALL_API_KEY="your_api_key_here"  # Get from football-data.org
DATABASE_URL="your_neon_postgres_url" # PostgreSQL connection string
```

*Note for Reviewers: The prediction API (`/predict`) includes graceful error handling. You can still test the ML inference locally even without a PostgreSQL connection string (database logging will simply be bypassed).*

## ⚙️ Running the Pipeline

Execute the pipeline sequentially to generate the features, train the model, and launch the API.

**1. Feature Engineering:**
Generate historical statistics, rolling form, and Elo ratings.

```bash
python features.py
```

**2. Model Training:**
Train the XGBoost model and export the `.joblib` artifacts.

```bash
python train.py
```

**3. Launch the API:**
Start the FastAPI server.

```bash
uvicorn api:app --reload
```

*(Optional) Fetch Live Matches:* Update the dataset with the latest finished matches from the real world.

```bash
python scripts/update_data.py
```

## 🚀 API Usage

The API accepts match parameters and returns the predicted outcome alongside the top three mathematical features driving the decision.

**Endpoint:** `http://127.0.0.1:8000/predict` (POST)

**Request:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "hometeam": "Liverpool",
  "awayteam": "Man United"
}'
```

**Example Response:**

```json
{
  "status": "successful",
  "hometeam": "Liverpool",
  "awayteam": "Man United",
  "prediction": "Liverpool wins",
  "explanation": {
    "awayteam": -0.7285,
    "home_elo": 0.5385,
    "hometeam": 0.5325
  }
}
```

Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs` while the server is running.