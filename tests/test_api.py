from fastapi.testclient import TestClient
from api import app

client =TestClient(app)

def test_predict_success():
    response= client.post("/predict",json={"hometeam":"Arsenal","awayteam":"Chelsea"})
    
    assert response.status_code==200
    
    data=response.json()
    assert data["status"]=="successful"
    assert data["hometeam"]=="Arsenal"
    assert "prediction" in data
    
def test_predict_missing_data():
    response=client.post("/predict",json={"hometeam":"Arsenal"})
    
    assert response.status_code==422    