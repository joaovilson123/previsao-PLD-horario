import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================
PISOS_REGULATORIOS = {
    2021: 55.70,
    2022: 55.70,
    2023: 69.04,
    2024: 61.07,
    2025: 68.41,
    2026: 71.10
}


def limpar_numeros(series: pd.Series) -> pd.Series:
    """
    Trata a numeração brasileira de strings numéricas:
    Remove pontos de milhar e substitui vírgulas por pontos decimais.
    """
    if series.dtype == 'object':
        return (series.astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float))
    return series


def filtrar_por_subsistema(df: pd.DataFrame, colunas_candidatas: list, alvo: str) -> pd.DataFrame:
    """
    Filtra um DataFrame garantindo o isolamento do submercado/subsistema desejado.
    """
    for col in colunas_candidatas:
        if col in df.columns:
            mask = df[col].astype(str).str.upper().str.contains(alvo.upper(), na=False)
            if mask.any():
                return df[mask].copy()
    return df


def carregar_e_processar_dados(
    subsistema_alvo: str = 'SUDESTE',
    anos: list = None,
    train_pct: float = 0.80,
    val_pct: float = 0.10
):
    """
    Executa o pipeline completo de ETL e Engenharia de Features.

    Retorna:
        X_train, y_train, X_val, y_val, X_test, y_test (DataFrames/Series do Pandas)
    """
    if anos is None:
        anos = ['2021', '2022', '2023', '2024', '2025', '2026']

    print(f"[ETL] Carregando bases de dados de {anos[0]} a {anos[-1]}...")

    # 1. Leitura dos Arquivos CSV
    df_pld_raw = pd.concat([pd.read_csv(f'pld_horario_{ano}.csv', sep=';') for ano in anos], ignore_index=True)
    df_ear_raw = pd.concat([pd.read_csv(f'EAR_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos], ignore_index=True)
    df_ena_raw = pd.concat([pd.read_csv(f'ENA_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos], ignore_index=True)
    df_carga_raw = pd.concat([pd.read_csv(f'CURVA_CARGA_{ano}.csv', sep=';') for ano in anos], ignore_index=True)

    # 2. Filtragem por Submercado Regional
    print(f"[ETL] Isolando submercado: {subsistema_alvo}...")
    cols_sub = ['nom_subsistema', 'id_subsistema', 'submercado', 'nom_submercado']

    df_pld_sub = filtrar_por_subsistema(df_pld_raw, cols_sub, subsistema_alvo)
    df_ear_sub = filtrar_por_subsistema(df_ear_raw, cols_sub, subsistema_alvo)
    df_ena_sub = filtrar_por_subsistema(df_ena_raw, cols_sub, subsistema_alvo)
    df_carga_sub = filtrar_por_subsistema(df_carga_raw, cols_sub, subsistema_alvo)

    # 3. Construção dos Índices Temporais (Tratando Formatos Mistos de Data)
    df_pld = pd.DataFrame()
    df_pld['Data_Hora'] = pd.to_datetime({
        'year': df_pld_sub['MES_REFERENCIA'] // 100,
        'month': df_pld_sub['MES_REFERENCIA'] % 100,
        'day': df_pld_sub['DIA'],
        'hour': df_pld_sub['HORA']
    })
    df_pld['PLD'] = df_pld_sub['PLD_HORA'].values
    df_pld.set_index('Data_Hora', inplace=True)

    df_ear = pd.DataFrame()
    df_ear['Data_Hora'] = pd.to_datetime(df_ear_sub['ear_data'], format='mixed', dayfirst=True)
    df_ear['EAR'] = limpar_numeros(df_ear_sub['ear_verif_subsistema_percentual']).values
    df_ear.set_index('Data_Hora', inplace=True)

    df_ena = pd.DataFrame()
    df_ena['Data_Hora'] = pd.to_datetime(df_ena_sub['ena_data'], format='mixed', dayfirst=True)
    df_ena['ENA'] = limpar_numeros(df_ena_sub['ena_armazenavel_regiao_percentualmlt']).values
    df_ena.set_index('Data_Hora', inplace=True)

    df_carga = pd.DataFrame()
    df_carga['Data_Hora'] = pd.to_datetime(df_carga_sub['din_instante'], format='mixed', dayfirst=True)
    df_carga['CARGA'] = limpar_numeros(df_carga_sub['val_cargaenergiahomwmed']).values
    df_carga.set_index('Data_Hora', inplace=True)

    # Remoção de duplicidades de índice
    df_pld = df_pld[~df_pld.index.duplicated(keep='first')]
    df_ear = df_ear[~df_ear.index.duplicated(keep='first')]
    df_ena = df_ena[~df_ena.index.duplicated(keep='first')]
    df_carga = df_carga[~df_carga.index.duplicated(keep='first')]

    # 4. Alinhamento em Grade Temporal Contínua e Blindagem de Vazamento
    data_inicio = df_pld.index.min()
    data_fim = df_pld.index.max()
    grade_horaria = pd.date_range(start=data_inicio, end=data_fim, freq='h', name='Data_Hora')

    df_xgb = pd.DataFrame(index=grade_horaria)
    df_xgb = df_xgb.join(df_pld, how='left')
    df_xgb = df_xgb.join(df_ear, how='left')
    df_xgb = df_xgb.join(df_ena, how='left')
    df_xgb = df_xgb.join(df_carga, how='left')

    # Blindagem contra vazamento de dados:
    # - ffill() em vez de interpolate() para proibir uso de dados do futuro.
    # - shift(24) em EAR e ENA para refletir a defasagem real de publicação do ONS.
    df_xgb['EAR'] = df_xgb['EAR'].ffill().shift(24)
    df_xgb['ENA'] = df_xgb['ENA'].ffill().shift(24)
    df_xgb['PLD'] = df_xgb['PLD'].ffill()
    df_xgb['CARGA'] = df_xgb['CARGA'].ffill()

    # 5. Engenharia de Features
    print("[ETL] Gerando variáveis calendárias, lags e médias móveis...")

    # A) Calendário
    df_xgb['Hora'] = df_xgb.index.hour
    df_xgb['Dia_da_Semana'] = df_xgb.index.dayofweek
    df_xgb['Mes'] = df_xgb.index.month
    df_xgb['Eh_Fim_de_Semana'] = df_xgb['Dia_da_Semana'].isin([5, 6]).astype(int)

    # B) Lags Temporais
    for lag in [1, 2, 3, 24, 168]:
        df_xgb[f'PLD_lag_{lag}h'] = df_xgb['PLD'].shift(lag)
        df_xgb[f'CARGA_lag_{lag}h'] = df_xgb['CARGA'].shift(lag)

    # C) Médias Móveis
    df_xgb['PLD_media_movel_24h'] = df_xgb['PLD'].shift(1).rolling(window=24).mean()
    df_xgb['CARGA_media_movel_24h'] = df_xgb['CARGA'].shift(1).rolling(window=24).mean()

    # D) Target: PLD da hora seguinte (t+1)
    df_xgb['PLD_Target'] = df_xgb['PLD'].shift(-1)

    # Limpeza final de NaNs gerados pelos atrasos/lags
    df_xgb.dropna(inplace=True)

    # 6. Separação de Matrizes e Divisão Cronológica Tríplice
    X = df_xgb.drop(columns=['PLD_Target', 'CARGA'])
    y = df_xgb['PLD_Target']

    total_amostras = len(df_xgb)
    idx_treino = int(total_amostras * train_pct)
    idx_val = int(total_amostras * (train_pct + val_pct))

    X_train, y_train = X.iloc[:idx_treino], y.iloc[:idx_treino]
    X_val, y_val = X.iloc[idx_treino:idx_val], y.iloc[idx_treino:idx_val]
    X_test, y_test = X.iloc[idx_val:], y.iloc[idx_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test