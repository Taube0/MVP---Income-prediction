import pytest
from model_loader import load_model
import os

def test_load_existing_model():
    model_name = 'best_model.pkl'
    assert os.path.exists(model_name), f"O arquivo de modelo {model_name} não existe."

    model = load_model(model_name)
    assert model is not None, "O modelo carregado é None"

def test_load_nonexistent_model():
    model_name = 'nonexistent_model.pkl'
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model(model_name)

    assert str(excinfo.value) == f"Modelo '{model_name}' não encontrado."
