import pytest
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_model_performance():
    # Carrega o modelo e os dados de teste
    model_pipeline = joblib.load('best_model.pkl')
    X_train_resampled, X_test_preprocessed, y_train_resampled, y_test, preprocessor = joblib.load('data.pkl')

    # Fazendo predições nos dados de teste
    y_pred_prob = model_pipeline.predict_proba(X_test_preprocessed)[:, 1]

    # Definindo o threshold para classificação
    threshold = 0.5
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Calculando as métricas de desempenho
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Definindo os thresholds aceitáveis
    min_accuracy = 0.8
    min_precision = 0.75
    min_recall = 0.75
    min_f1 = 0.75
    min_roc_auc = 0.85

    # Assertivas para verificar se o modelo atende aos thresholds
    assert accuracy >= min_accuracy, f"Acurácia {accuracy} abaixo do limiar aceitável {min_accuracy}"
    assert precision >= min_precision, f"Precisão {precision} abaixo do limiar aceitável {min_precision}"
    assert recall >= min_recall, f"Recall {recall} abaixo do limiar aceitável {min_recall}"
    assert f1 >= min_f1, f"F1 Score {f1} abaixo do limiar aceitável {min_f1}"
    assert roc_auc >= min_roc_auc, f"ROC AUC {roc_auc} abaixo do limiar aceitável {min_roc_auc}"
