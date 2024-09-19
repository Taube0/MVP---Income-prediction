import joblib
import os

def load_model(model_name='best_model.pkl'):
    """Carrega o modelo salvo por padrão como 'best_model.pkl', ou um específico se necessário."""
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Modelo '{model_name}' não encontrado.")
    
    model = joblib.load(model_name)
    return model

# Carrega automaticamente o melhor modelo por padrão
def load_best_model():
    """Função para carregar automaticamente o melhor modelo ('best_model.pkl')."""
    return load_model()

# Validação básica após o carregamento do modelo
if __name__ == "__main__":
    model = load_best_model()  # Carrega o melhor modelo automaticamente
    print("Modelo carregado com sucesso:", model)
