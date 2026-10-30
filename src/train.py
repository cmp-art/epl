import pandas as pd
import joblib
from xgboost import XGBClassifier

def train_and_save (path, model_path):
    df=pd.read_csv(path)
    
    df['hometeam']=df['hometeam'].astype('category')
    df['awayteam']=df['awayteam'].astype('category')
     
    
    x=df.drop(columns=['ftr_encoded'])
    y=df['ftr_encoded']
    
    model=XGBClassifier(n_estimators=100,random_state=42,learning_rate=0.1,enable_categorical=True)
    model.fit(x,y)
    print("complete training")
    
    joblib.dump(model,model_path)
    print(f"saved model to {model_path}")
    
    joblib.dump(x.columns.tolist(),"models/features.joblib")
    


    
    