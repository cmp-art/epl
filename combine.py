import pandas as pd 
import os
import glob

file_list=glob.glob('data/raw/*.csv')

df_list=[pd.read_csv(file) for file in file_list]
comb=pd.concat(df_list,ignore_index=True)





comb['Date']=pd.to_datetime(comb['Date'],format='mixed',dayfirst=True)
comb=comb.sort_values(by='Date')

os.makedirs('data',exist_ok=True)

comb.to_csv('data/historical_matches.csv' , index=False)
print('Data is successful combine')
