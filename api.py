from fastapi import FastAPI, HTTPException
from pydantic import BaseModel , field_validator
import pandas as pd
import joblib
import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import shap
import json

load_dotenv()
DATABASE_URL=os.getenv("DATABASE_URL")


app=FastAPI(title='Epl result prediction',version='1.0')

try:
    db_pool=pool.SimpleConnectionPool(1,20,DATABASE_URL)
except Exception as e:
    print(f"Failed to establish pool:{e}")
    db_pool=None    

rmodel=joblib.load('models/predict.joblib')
feat=joblib.load('models/features.joblib')

with open('models/latest_stats.json','r') as f:
    latest_stats=json.load(f)
    
explainer=shap.TreeExplainer(rmodel)

class Modelrequest(BaseModel):
    hometeam:str
    awayteam:str
    @field_validator('hometeam','awayteam',mode='before')
    @classmethod
    def normalize_team_names(cls, value:str)-> str:
        return value.strip().title()
    
@app.post('/predict')
def Prediction_match(match:Modelrequest):
    
    try:
        
        ht = match.hometeam
        at = match.awayteam
        
        # Look up current stats. If team is new, use baseline defaults
        ht_stats = latest_stats.get(ht, {'elo': 1500, 'form': 0, 'goals_scored': 0, 'goals_conceded': 0})
        at_stats = latest_stats.get(at, {'elo': 1500, 'form': 0, 'goals_scored': 0, 'goals_conceded': 0})
        
        # Reconstruct the DataFrame with all required features
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
        
        df=pd.DataFrame([input_data])
        
        
        df['hometeam']=df['hometeam'].astype('category')
        df['awayteam']=df['awayteam'].astype('category')
    
        df=df[feat]
    
        pred=int(rmodel.predict(df)[0])
        
        shap_value=explainer.shap_values(df)
        
        if isinstance(shap_value,list):
            pred_shap=shap_value[pred][0]
        else:
            pred_shap=shap_value[0,:,pred]    
            
        feat_impact={f:round(float(val),4) for f,val in zip(feat,pred_shap)}    
        top_reason=dict(sorted(feat_impact.items(),key= lambda x:abs(x[1]),reverse=True)[:3])
    
        if pred == 2:
           res=f"{input_data['hometeam']} wins"
        elif pred == 0:
           res=f"{input_data['awayteam']} wins"
        else:
           res='draw'
           
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
                # Always return the connection to the pool
                db_pool.putconn(conn)  
        
        return {
         'status':'successful',
         'hometeam':input_data['hometeam'],
         'awayteam':input_data['awayteam'],
         'prediction':res,
         'explanation':top_reason
        }               
    
    except HTTPException as e  :  
        raise HTTPException(status_code=404,detail=str(e))