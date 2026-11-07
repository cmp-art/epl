import os
import requests
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY:
    raise ValueError("FOOTBALL_API_KEY environment variable is missing or empty.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing or empty.")

TEAM_NAME_MAP = {
    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull",
    "Sunderland AFC": "Sunderland",
    "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham",
    "Leicester City FC": "Leicester",
    "Leeds United FC": "Leeds",
    "Norwich City FC": "Norwich",
    "Newcastle United FC": "Newcastle",
    "Manchester United FC":"Man United",
    "Manchester City FC":"Man City",
    "Brighton & Hove Albion":"Brighton",
    "Ipswich Town":"Ipswich"
    }

def clean_team_name(name):
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    name=name.replace(" FC","").replace(" AFC","").replace("AFC ","")
    return name.strip()

def fetch_latest_result():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
    headers = {"X-AUTH-TOKEN": API_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"response error: {response.text}")
    
    return response.json().get('matches', [])

def update_dataset_and_db(matches):
    new_data = []
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database connection failed: {e}")
        conn, cursor = None, None

    for match in matches:
        ht = clean_team_name(match['homeTeam']['name'])
        at = clean_team_name(match['awayTeam']['name'])
        h_score = match['score']['fullTime']['home']
        a_score = match['score']['fullTime']['away']
        
        new_data.append({
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG": h_score,
            "FTAG": a_score,
            "Date": match['utcDate']
        })
        
        if cursor:
            # Determine actual outcome format to match prediction strings
            if h_score > a_score: 
                actual = f"{ht} wins"
            elif a_score > h_score: 
                actual = f"{at} wins"
            else: 
                actual = 'draw'
                
            update_query = """
            UPDATE prediction_logs 
            SET actual_outcome = %s 
            WHERE hometeam = %s AND awayteam = %s AND actual_outcome IS NULL
            """
            cursor.execute(update_query, (actual, ht, at))
            
    if conn and cursor:
        conn.commit()
        cursor.close()
        conn.close()
        print("Database prediction logs updated.")
        
    new_df = pd.DataFrame(new_data)
    csv_path = 'data/historical_matches.csv'
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if the historical file exists before trying to read it
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['Date', 'HomeTeam'])
    else:
        print(f"{csv_path} not found. Creating a new one.")
        updated_df = new_df
        
    updated_df.to_csv(csv_path, index=False)
    print('EPL data updated successfully')
    
if __name__ == "__main__":
    latest_matches = fetch_latest_result()
    if latest_matches:
        update_dataset_and_db(latest_matches)