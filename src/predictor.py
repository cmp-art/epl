import pandas as pd 
import joblib
import json

def make_prediction(input_data):
    loaded_model = joblib.load('models/predict.joblib')
    exp_features = joblib.load('models/features.joblib')
    
    # Load the dynamic stats generated during training
    with open('models/latest_stats.json', 'r') as f:
        latest_stats = json.load(f)
        
    ht = input_data['hometeam']
    at = input_data['awayteam']
    
    # Fetch stats with fallbacks for newly promoted teams
    ht_stats = latest_stats.get(ht, {'elo': 1500, 'form': 0, 'goals_scored': 0, 'goals_conceded': 0})
    at_stats = latest_stats.get(at, {'elo': 1500, 'form': 0, 'goals_scored': 0, 'goals_conceded': 0})
    
    # Append engineered features to input_data
    input_data.update({
        'home_elo': ht_stats['elo'],
        'away_elo': at_stats['elo'],
        'home_form': ht_stats['form'],
        'away_form': at_stats['form'],
        'home_goals_scored': ht_stats['goals_scored'],
        'away_goals_scored': at_stats['goals_scored'],
        'home_goals_conceded': ht_stats['goals_conceded'],
        'away_goals_conceded': at_stats['goals_conceded']
    })
    
    df = pd.DataFrame([input_data])
    
    df['hometeam'] = df['hometeam'].astype('category')
    df['awayteam'] = df['awayteam'].astype('category')
    
    # Filter to exactly the features the model was trained on
    df = df[exp_features]
    
    pred = loaded_model.predict(df)[0]
    
    if pred == 2:
        res = f"{input_data['hometeam']} wins"
    elif pred == 0:
        res = f"{input_data['awayteam']} wins" 
    else:
        res = 'draw'
    print(f'match prediction: {res}')
    return res