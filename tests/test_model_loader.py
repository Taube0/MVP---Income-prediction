import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_loader import load_model


def test_load_existing_model():
    """
    Testa o carregamento de um modelo existente.
    
    Verifica:
    - Se o arquivo do modelo existe no caminho esperado.
    - Se o modelo pode ser carregado corretamente.
    """
    model_name = 'best_model.pkl'
    assert os.path.exists(model_name), f"O arquivo de modelo {model_name} não existe."

    model = load_model(model_name)
    assert model is not None, "O modelo carregado é None"

def test_load_nonexistent_model():
    """
    Testa o comportamento ao tentar carregar um modelo inexistente.
    
    Verifica:
    - Se uma exceção FileNotFoundError é levantada ao tentar carregar um modelo que não existe.
    """
    model_name = 'nonexistent_model.pkl'
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model(model_name)

    assert str(excinfo.value) == f"Modelo '{model_name}' não encontrado."
