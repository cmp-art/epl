import streamlit as st
import pandas as pd
import joblib
import json
import shap
import os

# --- SETTINGS & INITIALIZATION ---
st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽", layout="centered")

# Initialize ephemeral session history
if 'history' not in st.session_state:
    st.session_state.history = []

# --- LOAD ASSETS (Cached so they only load once) ---
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load('models/predict.joblib')
    explainer = shap.TreeExplainer(model)
    feat = joblib.load('models/features.joblib')
    return model, explainer, feat

@st.cache_data
def load_stats():
    with open('models/latest_stats.json', 'r') as f:
        return json.load(f)

try:
    rmodel, explainer, feat = load_model_and_explainer()
    latest_stats = load_stats()
    valid_teams = sorted(latest_stats.keys())
except Exception as e:
    st.error(f"Failed to load models or stats. Ensure the 'models' directory exists. Error: {e}")
    st.stop()

# --- UI FRONTEND ---
st.title("⚽ EPL Match Predictor")
st.write("Select teams to predict outcomes. History clears when you close this window.")

# Dropdown Selection
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", ["Select Team..."] + valid_teams)
with col2:
    away_team = st.selectbox("Away Team", ["Select Team..."] + valid_teams)

# Prediction Button
if st.button("Predict Match", type="primary", use_container_width=True):
    if home_team == "Select Team..." or away_team == "Select Team...":
        st.warning("⚠️ Please select both teams.")
    elif home_team == away_team:
        st.warning("⚠️ Home and Away teams must be different.")
    else:
        # --- PREDICTION LOGIC ---
        ht_stats = latest_stats[home_team]
        at_stats = latest_stats[away_team]
        
        input_data = {
            'hometeam': home_team,
            'awayteam': away_team,
            'home_elo': ht_stats['elo'],
            'away_elo': at_stats['elo'],
            'home_form': ht_stats['form'],
            'away_form': at_stats['form'],
            'home_goals_scored': ht_stats['goals_scored'],
            'away_goals_scored': at_stats['goals_scored'],
            'home_goals_conceded': ht_stats['goals_conceded'],
            'away_goals_conceded': at_stats['goals_conceded']
        }
        
        df = pd.DataFrame([input_data])
        
        # Enforce global categories
        df['hometeam'] = pd.Categorical(df['hometeam'], categories=valid_teams)
        df['awayteam'] = pd.Categorical(df['awayteam'], categories=valid_teams)
        df = df[feat]
        
        # Predict
        pred = int(rmodel.predict(df)[0])
        
        # SHAP Explanations
        shap_value = explainer.shap_values(df)
        if isinstance(shap_value, list):
            pred_shap = shap_value[pred][0]
        else:
            pred_shap = shap_value[0, :, pred]    
            
        feat_impact = {f: round(float(val), 4) for f, val in zip(feat, pred_shap)}    
        top_reason = dict(sorted(feat_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:1])
        top_factor_name = list(top_reason.keys())[0]
        
        if pred == 2:
            res = f"{home_team} Wins"
        elif pred == 0:
            res = f"{away_team} Wins"
        else:
            res = "Draw"
            
        # Add to ephemeral history
        st.session_state.history.insert(0, {
            "Home": home_team,
            "Away": away_team,
            "Prediction": res,
            "Key Factor": top_factor_name
        })
        
        st.success(f"**Predicted Outcome:** {res}")

# --- HISTORY TABLE ---
if st.session_state.history:
    st.divider()
    st.subheader("Current Session History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)