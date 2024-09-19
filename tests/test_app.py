import pytest
import json
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    # Entrada válida
    sample_input = {
        'features': [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married',
                     'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0,
                     40, 'United-States']
    }

    response = client.post('/predict', json=sample_input)
    assert response.status_code == 200

    data = response.get_json()
    assert 'prediction' in data
    assert data['prediction'] in [0, 1]

def test_predict_missing_features(client):
    # Dados faltando
    sample_input = {}

    response = client.post('/predict', json=sample_input)
    assert response.status_code == 400

    data = response.get_json()
    assert 'error' in data

def test_predict_invalid_data_types(client):
    # Tipos de dados inválidos
    sample_input = {
        'features': ['thirty-nine', 'State-gov', 'seventy seven thousand', 'Bachelors', 'thirteen', 'Never-married',
                     'Adm-clerical', 'Not-in-family', 'White', 'Male', 'two thousand one hundred seventy-four', 'zero',
                     'forty', 'United-States']
    }

    response = client.post('/predict', json=sample_input)
    assert response.status_code in [400, 500]

    data = response.get_json()
    assert 'error' in data
