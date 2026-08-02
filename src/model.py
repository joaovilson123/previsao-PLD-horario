import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

PLD_PISO = 61.07

def preparar_dados_e_treinar(df: pd.DataFrame):
    """Executa o corte cronológico, treina o XGBoost e avalia no conjunto de teste."""
    X = df.drop(columns=['PLD_Target', 'CARGA'])
    y = df['PLD_Target']

    total = len(df)
    idx_treino = int(total * 0.80)
    idx_val = int(total * 0.90)

    X_train, y_train = X.iloc[:idx_treino], y.iloc[:idx_treino]
    X_val, y_val     = X.iloc[idx_treino:idx_val], y.iloc[idx_treino:idx_val]
    X_test, y_test   = X.iloc[idx_val:], y.iloc[idx_val:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Predição e Trava Regulatória
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, PLD_PISO)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n--- MÉTRICAS DE AVALIAÇÃO ---")
    print(f"RMSE: {rmse:.2f} R$/MWh")
    print(f"MAE:  {mae:.2f} R$/MWh")

    return model, X_test, y_test, y_pred