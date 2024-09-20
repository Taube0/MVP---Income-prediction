import pytest
import json
from app import app

@pytest.fixture
def client():
    """Fixture para criar um cliente de teste Flask."""
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    """
    Testa a rota /predict com uma entrada válida.
    
    Verifica:
    - Se o status da resposta é 200 (OK).
    - Se o campo 'prediction' está presente na resposta.
    - Se o valor da predição é 0 ou 1 (valores válidos).
    """
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
    """
    Testa a rota /predict com dados faltando.
    
    Verifica:
    - Se o status da resposta é 400 (erro do cliente).
    - Se a resposta contém uma mensagem de erro indicando que os dados de entrada estão ausentes ou incorretos.
    """
    sample_input = {}

    response = client.post('/predict', json=sample_input)
    assert response.status_code == 400

    data = response.get_json()
    assert 'error' in data

def test_predict_invalid_data_types(client):
    """
    Testa a rota /predict com tipos de dados inválidos.
    
    Verifica:
    - Se o status da resposta é 400 ou 500, indicando que o servidor não conseguiu processar a entrada.
    - Se a resposta contém uma mensagem de erro apropriada.
    """
    sample_input = {
        'features': ['thirty-nine', 'State-gov', 'seventy seven thousand', 'Bachelors', 'thirteen', 'Never-married',
                     'Adm-clerical', 'Not-in-family', 'White', 'Male', 'two thousand one hundred seventy-four', 'zero',
                     'forty', 'United-States']
    }

    response = client.post('/predict', json=sample_input)
    assert response.status_code in [400, 500]

    data = response.get_json()
    assert 'error' in data
