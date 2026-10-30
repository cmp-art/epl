from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import psycopg2
from dotenv import load_dotenv
import shap

load_dotenv()
DATABASE_URL=os.getenv("DATABASE_URL")


app=FastAPI(title='Epl result prediction',version='1.0')

rmodel=joblib.load('models/predict.joblib')
feat=joblib.load('models/features.joblib')

explainer=shap.TreeExplainer(rmodel)

class Modelrequest(BaseModel):
    hometeam:str
    awayteam:str
    
@app.post('/predict')
def Prediction_match(match:Modelrequest):
    
    try:
        
        input_data=match.model_dump()
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
           
        try:
            
            conn=psycopg2.connect(DATABASE_URL)
            cursor=conn.cursor ()
        
            insert="""INSERT INTO prediction_logs(hometeam, awayteam, predicted_outcome)
            VALUES(%s,%s,%s)"""     
        
            cursor.execute(insert,(input_data['hometeam'],input_data['awayteam'],res)) 
        
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_error: 
           print (f'Failed to log to database {db_error}')   
        
        return {
         'status':'successful',
         'hometeam':input_data['hometeam'],
         'awayteam':input_data['awayteam'],
         'prediction':res,
         'explanation':top_reason
        }               
    
    except HTTPException as e  :  
        raise HTTPException(status_code=404,detail=str(e))