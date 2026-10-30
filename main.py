from src.dataloader import load_and_clean
from src.features import generate_match_features
from src.train import train_and_save
from src.predictor  import make_prediction
import joblib

if __name__=="__main__":
 load_and_clean(file='data/results.csv')

 generate_match_features(d='data/clean.csv')

 train_and_save(path='data/feature.csv',model_path='models/predict.joblib')
 
 sample={
     'hometeam':'Chelsea',
     'awayteam':'Arsenal'
 }
 
 make_prediction(input_data=sample)

 feat =joblib.load('models/features.joblib')
 print(f'required features: {feat}')