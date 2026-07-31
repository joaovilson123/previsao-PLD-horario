import pandas as pd
from pathlib import Path
# Encontra a pasta raiz do projeto (Sobe 1 nível a partir do diretório do próprio script)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"


def limpar_numeros(series: pd.Series) -> pd.Series:
    """Trata numeração brasileira (remove pontos de milhar, troca vírgula por ponto)."""
    if series.dtype == 'object':
        return (series.astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float))
    return series


def carregar_e_unificar_dados(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Carrega os CSVs brutos de 2021 a 2025 e constrói a base unificada temporalmente."""
    anos = range(2021, 2026)

    # Carregamento
    pld_files = [pd.read_csv(f'{data_dir}/pld_horario_{ano}.csv', sep=';') for ano in anos]
    ear_files = [pd.read_csv(f'{data_dir}/EAR_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos]
    ena_files = [pd.read_csv(f'{data_dir}/ENA_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos]
    carga_files = [pd.read_csv(f'{data_dir}/CURVA_CARGA_{ano}.csv', sep=';') for ano in anos]

    df_pld_raw = pd.concat(pld_files, ignore_index=True)
    df_ear_raw = pd.concat(ear_files, ignore_index=True)
    df_ena_raw = pd.concat(ena_files, ignore_index=True)
    df_carga_raw = pd.concat(carga_files, ignore_index=True)

    # Processamento dos Índices
    df_pld = pd.DataFrame({
        'Data_Hora': pd.to_datetime({
            'year': df_pld_raw['MES_REFERENCIA'] // 100,
            'month': df_pld_raw['MES_REFERENCIA'] % 100,
            'day': df_pld_raw['DIA'],
            'hour': df_pld_raw['HORA']
        }),
        'PLD': df_pld_raw['PLD_HORA']
    }).set_index('Data_Hora')

    df_ear = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ear_raw['ear_data'], format='%d/%m/%Y'),
        'EAR': limpar_numeros(df_ear_raw['ear_verif_subsistema_percentual'])
    }).set_index('Data_Hora')

    df_ena = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ena_raw['ena_data'], format='%d/%m/%Y'),
        'ENA': limpar_numeros(df_ena_raw['ena_armazenavel_regiao_percentualmlt'])
    }).set_index('Data_Hora')

    df_carga = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_carga_raw['din_instante'], format='%d/%m/%Y %H:%M'),
        'CARGA': limpar_numeros(df_carga_raw['val_cargaenergiahomwmed'])
    }).set_index('Data_Hora')

    # Merge e Interpolação
    df_merged = df_pld.join([df_ear, df_ena, df_carga], how='outer').sort_index()
    df_merged['EAR'] = df_merged['EAR'].ffill()
    df_merged['ENA'] = df_merged['ENA'].ffill()
    df_merged.dropna(subset=['PLD', 'CARGA'], inplace=True)

    return df_merged