import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importação do módulo local de carregamento de dados e constantes
from data_loader import carregar_e_processar_dados, PISOS_REGULATORIOS

# Configuração visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')


def treinar_e_avaliar_modelo(subsistema_alvo: str = 'SUDESTE'):
    """
    Pipeline principal de treinamento, predição, pós-processamento e plotagem.
    """
    # ==========================================================================
    # 1. OBTENÇÃO DOS DADOS TRATADOS
    # ==========================================================================
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_e_processar_dados(
        subsistema_alvo=subsistema_alvo
    )

    print("\n" + "=" * 50)
    print(f"DIVISÃO DOS DADOS DE TREINAMENTO ({subsistema_alvo})")
    print("=" * 50)
    print(f"Treino:    {len(X_train)} amostras (80%)")
    print(f"Validação: {len(X_val)} amostras (10%)")
    print(f"Teste:     {len(X_test)} amostras (10%) - [Totalmente cego]")

    # ==========================================================================
    # 2. TREINAMENTO DO XGBOOST
    # ==========================================================================
    print("\nIniciando o treinamento do modelo XGBoost...")

    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30
    )

    # Parada antecipada avaliada estritamente no conjunto de validação
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # ==========================================================================
    # 3. PREDIÇÃO E APLICAÇÃO DO PISO REGULATÓRIO DINÂMICO
    # ==========================================================================
    print("\nRealizando predições no conjunto de teste...")
    y_pred = model_xgb.predict(X_test)

    # Mapeamento do piso correspondente ao ano de cada registro do teste
    anos_teste = y_test.index.year
    piso_dinamico = anos_teste.map(PISOS_REGULATORIOS).values
    y_pred_post = np.maximum(y_pred, piso_dinamico)

    # ==========================================================================
    # 4. AVALIAÇÃO DE MÉTRICAS
    # ==========================================================================
    mse = mean_squared_error(y_test, y_pred_post)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_post)

    print("\n" + "=" * 50)
    print("MÉTRICAS DE DESEMPENHO REAL (SEM VAZAMENTO DE DADOS)")
    print("=" * 50)
    print(f"MSE  (Erro Quadrático Médio):       {mse:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático):    {rmse:.2f} R$/MWh")
    print(f"MAE  (Erro Médio Absoluto):        {mae:.2f} R$/MWh")

    """

    # ==========================================================================
    # 5. VISUALIZAÇÃO GRÁFICA
    # ==========================================================================
    inicio, fim = 0, min(1000, len(y_test))

    # Gráfico 1: Aderência Temporal (Real vs. Previsto)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        y_test.index[inicio:fim], y_test.iloc[inicio:fim],
        label='PLD Real (R$/MWh)', color='#2ca02c', linewidth=2, alpha=0.85
    )
    ax1.plot(
        y_test.index[inicio:fim], y_pred_post[inicio:fim],
        label='PLD Previsto pelo XGBoost', color='#d62728', linewidth=1.8, linestyle='--'
    )
    ax1.set_title(
        f"Aderência Temporal (Submercado {subsistema_alvo}): Real vs. Previsto",
        fontsize=14, fontweight='bold', pad=12
    )
    ax1.set_ylabel("Preço de Liquidação - PLD (R$/MWh)", fontsize=11)
    ax1.set_xlabel("Data / Hora", fontsize=11)
    ax1.legend(loc='upper right', fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Importância das Variáveis (Feature Importance)
    fig, ax2 = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(
        model_xgb, max_num_features=12, importance_type='gain',
        ax=ax2, color='#1f77b4', show_values=False
    )
    ax2.set_title("Top 12 Variáveis Mais Importantes (Gain)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Ganho Relativo (Importance Gain)", fontsize=11)
    ax2.set_ylabel("Atributos (Features)", fontsize=11)
    plt.tight_layout()
    plt.show()

"""
if __name__ == '__main__':
    # Executa a pipeline para o subsistema definido
    treinar_e_avaliar_modelo(subsistema_alvo='SUDESTE')