# ⚽ EPL Match Predictor: Executive & Operational Guide

🚀 **Live Application:** [https://ebgavghk8crz2d935h6jpf.streamlit.app/](https://ebgavghk8crz2d935h6jpf.streamlit.app/)

## 📌 Project Overview

The **EPL Match Predictor** is an intelligent analytics tool designed to forecast English Premier League match outcomes. Built using machine learning (**XGBoost**), it evaluates team strengths using historical match data, dynamic **Elo ratings**, and recent team form.

Instead of dealing with complex code or command-line scripts, managers and analysts can interact with the system through a clean, web-based dashboard built with **Streamlit**.

## 🚀 Key Features for Managers

* **Interactive Selection:** Dropdown menus built directly from live team databases, ensuring zero typo errors when selecting fixtures.
* **Instant Predictions:** Forecasts whether the Home team wins, Away team wins, or if the match ends in a draw.
* **Transparent Logic (SHAP):** Highlights the primary driving factors (e.g., Elo rating gaps, goalscoring form, or away-team disadvantages) behind every prediction, turning a "black box" AI into an interpretable tool.
* **Secure & Ephemeral History:** Automatically tracks all match predictions run during the active session. The history clears instantly the moment the browser tab is closed.

## 🛠️ Project Structure at a Glance

* `app.py`: The interactive web dashboard you open in your browser.
* `update_data.py`: Fetches the latest real-world match results and updates standings.
* `models/`: Contains the trained AI model (`predict.joblib`), feature lists, and team statistics (`latest_stats.json`).
* `src/`: Contains the backend data cleaning and feature engineering engines (`dataloader.py`, `features.py`, `train.py`).

---

## 💻 How to Run the App Locally (Step-by-Step)

If you prefer to run the application on your own machine rather than the cloud link, follow these simple steps using your computer's terminal:

### Step 1: Open the Project Folder & Setup

Open your terminal (Command Prompt, PowerShell, or Terminal), navigate to the project directory, and install dependencies (only needed the first time):

```bash
cd path/to/epl
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

Activate Python's virtual environment where all dependencies are safely stored:

* **On Windows (PowerShell):**
  ```bash
  .venv\Scripts\Activate
  ```

* **On Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

*(Once activated, run `pip install -r requirements.txt` if you haven't installed the packages yet).*

### Step 3: Launch the Dashboard

Run the Streamlit application with a single command:
```bash
streamlit run app.py
```

Your default web browser will automatically open a window displaying the interactive dashboard (`http://localhost:8501`).

---

## 🔄 Updating Data & Retraining the Model

If a new match week has passed and you want the AI to incorporate the freshest real-world results into its calculations, run the complete data and training pipeline sequentially before launching the app. 

*Make sure your virtual environment is activated before running these!*

1. **Fetch latest match results:**
   ```bash
   python update_data.py
   ```

2. **Clean the raw data:**
   ```bash
   python src/dataloader.py
   ```

3. **Recompute team metrics and Elo ratings:**
   ```bash
   python src/features.py
   ```

4. **Retrain the AI model:**
   ```bash
   python src/train.py
   ```

Once Step 4 says "complete training", the new model is saved. You can then run `streamlit run app.py` (or push the code to GitHub to update your live app) and enjoy the freshest predictions!