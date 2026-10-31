import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv("FOOTBALL_API_KEY")

def fetch_latest_result():
    url ="https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
    headers={"X-AUTH-TOKEN":API_KEY}
    
    response=requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"response error: {response.text}")
    
    matches=response.json().get('matches',[])
    
    new_data=[]
    for match in matches:
        new_data.append({
            "hometeam":match['hometeam']['name'],
            "awayteam":match['awayteam']['name'],
            "homescore":match['score']['fullTime']['home'],
            "awayscore":match['score']['fullTime']['away'],
            "date":match['utcDate']
        })
        
    return pd.DataFrame(new_data)    

def update_dataset(new_df):
    existing_df=pd.read_csv('data/historical_matches.csv')
    
    updated_df=pd.concat([existing_df,new_df]).drop_duplicates(subset=['date','hometeam'])
    updated_df.to_csv('data/historical_matches.csv',index=False)
    print('Epl data updated successfull')
    
if __name__ == "__main__":
    latest_matches=fetch_latest_result()
    if not latest_matches.empty:
        update_dataset(latest_matches)   