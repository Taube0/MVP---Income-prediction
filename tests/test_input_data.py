import pytest
import pandas as pd
import numpy as np

def test_input_data_format():
    # Definir os nomes das colunas conforme o dataset original
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country']

    # Exemplo de dados de entrada
    features = [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married',
                'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0,
                40, 'United-States']

    # Verifica se o número de features está correto
    assert len(features) == len(columns), "Número de features não corresponde ao esperado."

    # Cria DataFrame e verifica os tipos
    features_df = pd.DataFrame([features], columns=columns)

    expected_dtypes = {
        'age': np.number,
        'workclass': object,
        'fnlwgt': np.number,
        'education': object,
        'education-num': np.number,
        'marital-status': object,
        'occupation': object,
        'relationship': object,
        'race': object,
        'sex': object,
        'capital-gain': np.number,
        'capital-loss': np.number,
        'hours-per-week': np.number,
        'native-country': object
    }

    for column, expected_dtype in expected_dtypes.items():
        assert features_df[column].dtype == expected_dtype, f"O tipo da coluna {column} deve ser {expected_dtype}."
