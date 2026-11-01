import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import os

def train_and_save (path, model_path):
    df=pd.read_csv(path)
    
    df['hometeam']=df['hometeam'].astype('category')
    df['awayteam']=df['awayteam'].astype('category')
     
    
    x=df.drop(columns=['ftr_encoded'])
    y=df['ftr_encoded']
    
    x=x.select_dtypes(exclude=['object','str'])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    
    model=XGBClassifier(n_estimators=100,random_state=42,learning_rate=0.1,enable_categorical=True)
    model.fit(x_train,y_train,eval_set=[(x_test,y_test)],verbose=10)
    print("complete training")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs("models",exist_ok=True)
    
    joblib.dump(model,model_path)
    print(f"saved model to {model_path}")
    
    joblib.dump(x_train.columns.tolist(),"models/features.joblib")
    
if __name__=="__main__":
    train_and_save('data/feature.csv', 'models/predict.joblib')    
    


    
    