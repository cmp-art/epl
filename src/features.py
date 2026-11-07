import pandas as pd 
import numpy as np
import json

def generate_match_features(d):
    df = pd.read_csv(d)
    df = df.sort_values(by='datetime').reset_index(drop=True)
    
    
    if 'ftr' not in df.columns:
        df['ftr'] = np.nan
        
    def determine_ftr(row):
       
        if pd.notna(row.get('ftr')) and str(row.get('ftr')).strip().upper() in ['A', 'H', 'D']:
            return str(row.get('ftr')).strip().upper()
        
        
        if row['fthg'] > row['ftag']:
            return 'H'
        elif row['ftag'] > row['fthg']:
            return 'A'
        else:
            return 'D'
            
    df['ftr'] = df.apply(determine_ftr, axis=1)
    
    # 2. CALCULATE ELO AND FORM
    elo_dict = {}
    form_tracker = {}
    
    home_elo, away_elo = [], []
    home_form, away_form = [], []
    home_goals_scored, away_goals_scored = [], []
    home_goals_conceded, away_goals_conceded = [], []
    
    for index, row in df.iterrows():
        ht = row['hometeam']
        at = row['awayteam'] 
        
        if ht not in elo_dict:
            elo_dict[ht] = 1500
            form_tracker[ht] = {'points':[], 'scored':[], 'conceded':[]}
        if at not in elo_dict:
            elo_dict[at] = 1500
            form_tracker[at] = {'points':[], 'scored':[], 'conceded':[]}
            
        home_elo.append(elo_dict[ht])
        away_elo.append(elo_dict[at])
        
        home_form.append(np.mean(form_tracker[ht]['points'][-5:]) if form_tracker[ht]['points'] else 0)
        away_form.append(np.mean(form_tracker[at]['points'][-5:]) if form_tracker[at]['points'] else 0)
        
        home_goals_scored.append(np.mean(form_tracker[ht]['scored'][-5:]) if form_tracker[ht]['scored'] else 0)
        away_goals_scored.append(np.mean(form_tracker[at]['scored'][-5:]) if form_tracker[at]['scored'] else 0)
        
        home_goals_conceded.append(np.mean(form_tracker[ht]['conceded'][-5:]) if form_tracker[ht]['conceded'] else 0)
        away_goals_conceded.append(np.mean(form_tracker[at]['conceded'][-5:]) if form_tracker[ht]['conceded'] else 0)
        
        ftr = row['ftr']
        h_goals, a_goals = row['fthg'], row['ftag']
        
        h_pts = 3 if ftr == 'H' else (1 if ftr == 'D' else 0)
        a_pts = 3 if ftr == 'A' else (1 if ftr == 'D' else 0)
        
        form_tracker[ht]['points'].append(h_pts)
        form_tracker[ht]['scored'].append(h_goals)
        form_tracker[ht]['conceded'].append(a_goals)
        form_tracker[at]['points'].append(a_pts)
        form_tracker[at]['scored'].append(a_goals)
        form_tracker[at]['conceded'].append(h_goals)
        
        dr = elo_dict[ht] - elo_dict[at] + 100
        expected_h = 1 / (1 + 10**(-dr/100))
        expected_a = 1 / (1 + 10**(dr/100))
        
        sa = 1 if ftr == 'H' else (0.5 if ftr == 'D' else 0)
        sb = 1 if ftr == 'A' else (0.5 if ftr == 'D' else 0)
        
        elo_dict[ht] += 20 * (sa - expected_h)
        elo_dict[at] += 20 * (sb - expected_a)
        
    df['home_elo'] = home_elo
    df['away_elo'] = away_elo
    df['home_form'] = home_form
    df['away_form'] = away_form
    df['home_goals_scored'] = home_goals_scored
    df['away_goals_scored'] = away_goals_scored
    df['home_goals_conceded'] = home_goals_conceded
    df['away_goals_conceded'] = away_goals_conceded
    
    # 3. SAVE LATEST STATS
    latest_stats = {}
    for team in elo_dict.keys():
        latest_stats[team] = {
            'elo': elo_dict[team],
            'form': np.mean(form_tracker[team]['points'][-5:]) if form_tracker[team]['points'] else 0,
            'goals_scored': np.mean(form_tracker[team]['scored'][-5:]) if form_tracker[team]['scored'] else 0,
            'goals_conceded': np.mean(form_tracker[team]['conceded'][-5:]) if form_tracker[team]['conceded'] else 0
        }
    with open('models/latest_stats.json', 'w') as f:
        json.dump(latest_stats, f)
        
    # 4. ENCODE AND CLEANUP
    df['ftr_encoded'] = df['ftr'].map({'A': 0, 'H': 2, 'D': 1})
   
    keep_columns = [
        'hometeam', 'awayteam', 'home_elo', 'away_elo', 
        'home_form', 'away_form', 'home_goals_scored', 
        'away_goals_scored', 'home_goals_conceded', 
        'away_goals_conceded', 'ftr_encoded'
    ]
    df_new = df[keep_columns]
    
    print('Featuring done with dynamic metrics')
    df_new.to_csv('data/feature.csv', index=False)
    
    return df_new        
        
if __name__ == "__main__": 
    generate_match_features('data/clean.csv')