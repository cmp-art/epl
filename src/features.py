import pandas as pd 
import numpy as np

def generate_match_features(d) :
    
    df=pd.read_csv(d)
   
    team_df=df[['hometeam','awayteam']]
    # target=df['ftr']
    
    df['ftr_encoded']=df['ftr'].map({'A':0,'H':2,'D':1})
    
    drop_f=['fthg', 'ftag', 'hthg', 'htag','hs', 'as', 'hst', 'ast', 'hr', 'ar','hometeam','ftr','awayteam','hy', 'htr']
    df_new=df.drop(columns=drop_f,errors='ignore')
    
    df_dumm=pd.get_dummies(df_new,drop_first=True,dtype=int)
    
    #df_dumm=pd.concat([df_dumm,target],axis=1)
    
    corr_mat=df_dumm.corr()['ftr_encoded'].abs()
    relevant_col=corr_mat[corr_mat>0.10].index.to_list()
    df_clear=df_dumm[relevant_col]
    
    
    correlat_mat=df_clear.corr().abs()
    upper=correlat_mat.where(np.triu(np.ones(correlat_mat.shape),k=1).astype(bool))
    reduct=[col for col in upper.columns if any(upper[col]>0.85)]
    df_clear=df_clear.drop(columns=reduct)
    
    df_clear=pd.concat([df_clear,team_df],axis=1)
    
    print('featuring done')
    
    df_clear.to_csv('data/feature.csv',index=False)
    
    return df_clear
    
    
    