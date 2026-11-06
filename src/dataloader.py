import pandas as pd
import numpy as np

def load_and_clean(file) :
    df=pd.read_csv(file,encoding='latin1')
    
    num_col=df.select_dtypes(include=[np.number]).columns
    
    empty = df.isnull().sum()
    
    print(f"null values: {empty}")
    
    for col in num_col:
        if df[col].isna().sum() > 0:
            df[col]=df[col].fillna(0)
            
    df=df.dropna()
    
    df.columns=[i.strip().lower().replace(" ","_") for i in df.columns ] 
    
    if 'date' in df.columns:
        df.rename(columns={'date':'datetime'}, inplace=True)
        
    df['datetime']=pd.to_datetime(df['datetime'],format='mixed',dayfirst=True) 
    print("cleaning done")
    df.to_csv("data/clean.csv",index=False) 
    
    return df    

if __name__ == "__main__":
    load_and_clean('data/historical_matches.csv')