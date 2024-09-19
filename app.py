import logging
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  # Adicionar pandas para trabalhar com DataFrames
import joblib
from flask import render_template

# Função para prever a classe com base na probabilidade e threshold
def predict_with_threshold(model, data, threshold=0.6):
    """Função para prever a classe com base na probabilidade e threshold."""
    probas = model.predict_proba(data)[:, 1]  # Pegamos a probabilidade da classe 1
    return (probas >= threshold).astype(int)

# Configurando o nível de log para DEBUG
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Carregando o pipeline completo (pré-processador + modelo)
model_pipeline = joblib.load('best_model.pkl')

# Carregando o pré-processador salvo no data.pkl
X_train_resampled, X_test_preprocessed, y_train_resampled, y_test, preprocessor = joblib.load('data.pkl')

# Definir os nomes das colunas conforme o dataset original
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country']

@app.route('/')
def index():
    """Rota para exibir a página inicial."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Rota para processar a predição dos dados recebidos."""
    app.logger.debug("Rota /predict acessada")
    try:
        data = request.json
        app.logger.debug(f"Dados recebidos: {data}")
        
        # Verifica se os dados foram enviados corretamente
        if not data or 'features' not in data:
            app.logger.error("Dados de entrada ausentes ou incorretos")
            return jsonify({'error': 'Dados de entrada ausentes ou incorretos'}), 400

        # Convertendo a entrada para um DataFrame
        features = np.array(data['features']).reshape(1, -1)
        features_df = pd.DataFrame(features, columns=columns)  # Criar DataFrame com as colunas corretas
        app.logger.debug(f"Features DataFrame: {features_df}")

        # Aplicando o pré-processador nos dados de entrada
        processed_features = preprocessor.transform(features_df)
        app.logger.debug(f"Features após o pré-processamento: {processed_features}")

        # Fazendo a predição com o pipeline (modelo já treinado)
        prediction = predict_with_threshold(model_pipeline, processed_features, threshold=0.4)
        app.logger.debug(f"Predição realizada: {prediction}")

        # Retorna a predição como resposta JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        app.logger.error(f"Erro ao processar a predição: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.logger.debug("Iniciando a aplicação Flask")
    app.run(debug=True)
