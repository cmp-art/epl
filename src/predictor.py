import pandas as pd 
import joblib

def make_prediction(input_data):
    loaded_model=joblib.load('models/predict.joblib')
    exp_features=joblib.load('models/features.joblib')
    
    df=pd.DataFrame([input_data])
    
    df['hometeam']=df['hometeam'].astype('category')
    df['awayteam']=df['awayteam'].astype('category')
    
    df=df[exp_features]
    
    pred =loaded_model.predict(df)[0]
    
    if pred==2:
        res=f"{input_data['hometeam']} wins"
    elif pred==0:
        res=f"{input_data['awayteam']} wins" 
    else:
        res='draw'
    print(f'match prediction: {res}')