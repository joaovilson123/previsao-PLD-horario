import pandas as pd


def criar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Gera calendários, lags temporais, médias móveis e define a variável alvo."""
    df_feat = df.copy()

    # Atributos Calendários
    df_feat['Hora'] = df_feat.index.hour
    df_feat['Dia_da_Semana'] = df_feat.index.dayofweek
    df_feat['Mes'] = df_feat.index.month
    df_feat['Eh_Fim_de_Semana'] = df_feat['Dia_da_Semana'].isin([5, 6]).astype(int)

    # Lags
    for lag in [1, 2, 3, 24, 168]:
        df_feat[f'PLD_lag_{lag}h'] = df_feat['PLD'].shift(lag)
        df_feat[f'CARGA_lag_{lag}h'] = df_feat['CARGA'].shift(lag)

    # Médias Móveis sem contaminação
    df_feat['PLD_media_movel_24h'] = df_feat['PLD'].shift(1).rolling(window=24).mean()
    df_feat['CARGA_media_movel_24h'] = df_feat['CARGA'].shift(1).rolling(window=24).mean()

    # Variável Alvo (t+1)
    df_feat['PLD_Target'] = df_feat['PLD'].shift(-1)

    # Remove instâncias nulas geradas pelos atrasos
    df_feat.dropna(inplace=True)
    return df_feat