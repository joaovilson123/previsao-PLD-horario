from src.data_loader import carregar_e_unificar_dados
from src.features import criar_features_temporais
from src.model import preparar_dados_e_treinar

if __name__ == "__main__":
    print("1. Carregando dados...")
    df_raw = carregar_e_unificar_dados()

    print("2. Gerando engenharia de recursos...")
    df_feat = criar_features_temporais(df_raw)

    print("3. Treinando modelo e avaliando...")
    model, X_test, y_test, y_pred = preparar_dados_e_treinar(df_feat)