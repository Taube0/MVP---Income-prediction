import pytest
import joblib
from sklearn.metrics import f1_score

def test_model_with_different_thresholds():
    model_pipeline = joblib.load('best_model.pkl')
    X_train_resampled, X_test_preprocessed, y_train_resampled, y_test, preprocessor = joblib.load('data.pkl')

    thresholds = [0.3, 0.5, 0.7]
    min_f1_scores = [0.60, 0.65, 0.65]  # Thresholds mínimos para cada valor de threshold

    for idx, threshold in enumerate(thresholds):
        y_pred_prob = model_pipeline.predict_proba(X_test_preprocessed)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        assert f1 >= min_f1_scores[idx], f"F1 Score {f1} abaixo do limiar aceitável {min_f1_scores[idx]} para o threshold {threshold}"
