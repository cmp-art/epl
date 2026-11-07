from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
import joblib
import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import shap
import json

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title='EPL Result Prediction', version='1.0')

try:
    db_pool = pool.SimpleConnectionPool(1, 20, DATABASE_URL)
except Exception as e:
    print(f"Failed to establish pool: {e}")
    db_pool = None    

# Load models and stats
rmodel = joblib.load('models/predict.joblib')
feat = joblib.load('models/features.joblib')

with open('models/latest_stats.json', 'r') as f:
    latest_stats = json.load(f)
    
explainer = shap.TreeExplainer(rmodel)

# Create a master list of all known teams from our features pipeline
valid_teams = list(latest_stats.keys())

class Modelrequest(BaseModel):
    hometeam: str
    awayteam: str
    
    @field_validator('hometeam', 'awayteam', mode='before')
    @classmethod
    def normalize_team_names(cls, value: str) -> str:
        return value.strip().title()
    
@app.post('/predict')
def Prediction_match(match: Modelrequest):
    try:
        ht = match.hometeam
        at = match.awayteam
        
        # 1. VERIFY TEAMS EXIST IN DATASET (Prevents 500 Server Errors on bad spelling)
        if ht not in valid_teams:
            raise HTTPException(status_code=400, detail=f"Home team '{ht}' not found. Check spelling. Valid teams include: {', '.join(valid_teams[:10])}...")
        if at not in valid_teams:
            raise HTTPException(status_code=400, detail=f"Away team '{at}' not found. Check spelling.")
        
        ht_stats = latest_stats[ht]
        at_stats = latest_stats[at]
        
        input_data = {
            'hometeam': ht,
            'awayteam': at,
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
    
        # 2. ENFORCE GLOBAL CATEGORIES (Fixes the XGBoost new team crash)
        df['hometeam'] = pd.Categorical(df['hometeam'], categories=valid_teams)
        df['awayteam'] = pd.Categorical(df['awayteam'], categories=valid_teams)
    
        df = df[feat]
    
        pred = int(rmodel.predict(df)[0])
        
        shap_value = explainer.shap_values(df)
        
        if isinstance(shap_value, list):
            pred_shap = shap_value[pred][0]
        else:
            pred_shap = shap_value[0, :, pred]    
            
        feat_impact = {f: round(float(val), 4) for f, val in zip(feat, pred_shap)}    
        top_reason = dict(sorted(feat_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    
        if pred == 2:
           res = f"{input_data['hometeam']} wins"
        elif pred == 0:
           res = f"{input_data['awayteam']} wins"
        else:
           res = 'draw'
           
        if db_pool:
            try:
                conn = db_pool.getconn()
                cursor = conn.cursor()
                insert = """INSERT INTO prediction_logs(hometeam, awayteam, predicted_outcome)
                            VALUES(%s,%s,%s)"""     
                cursor.execute(insert, (input_data['hometeam'], input_data['awayteam'], res)) 
                conn.commit()
                cursor.close()
            except Exception as db_error: 
                print(f'Failed to log to database {db_error}')
            finally:
                db_pool.putconn(conn)  
        
        return {
         'status': 'successful',
         'hometeam': input_data['hometeam'],
         'awayteam': input_data['awayteam'],
         'prediction': res,
         'explanation': top_reason
        }               
    
    except HTTPException as e:  
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))