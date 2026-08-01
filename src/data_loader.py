from pathlib import Path
import os
import pandas as pd
import requests

# Define diretórios base a partir da localização do script
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PARQUET_PATH = PROCESSED_DATA_DIR / "pld_historico.parquet"

# Mapeamento das URLs públicas do ONS
ONS_URLS = {
    'EAR': 'https://dados.ons.org.br/dataset/ear-diario-subsistema/resource/EAR_DIARIO_SUBSISTEMA_{ano}.csv',
    'ENA': 'https://dados.ons.org.br/dataset/ena-diario-subsistema/resource/ENA_DIARIO_SUBSISTEMA_{ano}.csv',
    'CARGA': 'https://dados.ons.org.br/dataset/curva-carga-2/resource/CURVA_CARGA_{ano}.csv'
}


def limpar_numeros(series: pd.Series) -> pd.Series:
    """Trata numeração brasileira (remove pontos de milhar, troca vírgula por ponto)."""
    if series.dtype == 'object':
        return (series.astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float))
    return series


def processar_e_unificar_dfs(pld_files: list, ear_files: list, ena_files: list, carga_files: list) -> pd.DataFrame:
    """Aplica tratamento de tipos, parsing de datas e merge entre os datasets."""
    df_pld_raw = pd.concat(pld_files, ignore_index=True)
    df_ear_raw = pd.concat(ear_files, ignore_index=True)
    df_ena_raw = pd.concat(ena_files, ignore_index=True)
    df_carga_raw = pd.concat(carga_files, ignore_index=True)

    # 1. PLD Horário
    df_pld = pd.DataFrame({
        'Data_Hora': pd.to_datetime({
            'year': df_pld_raw['MES_REFERENCIA'] // 100,
            'month': df_pld_raw['MES_REFERENCIA'] % 100,
            'day': df_pld_raw['DIA'],
            'hour': df_pld_raw['HORA']
        }),
        'PLD': limpar_numeros(df_pld_raw['PLD_HORA'])
    }).set_index('Data_Hora')

    # 2. EAR Diário
    col_data_ear = 'ear_data' if 'ear_data' in df_ear_raw.columns else df_ear_raw.columns[0]
    col_val_ear = 'ear_verif_subsistema_percentual' if 'ear_verif_subsistema_percentual' in df_ear_raw.columns else \
    df_ear_raw.columns[-1]

    df_ear = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ear_raw[col_data_ear], format='mixed'),
        'EAR': limpar_numeros(df_ear_raw[col_val_ear])
    }).drop_duplicates(subset=['Data_Hora']).set_index('Data_Hora')

    # 3. ENA Diário
    col_data_ena = 'ena_data' if 'ena_data' in df_ena_raw.columns else df_ena_raw.columns[0]
    col_val_ena = 'ena_armazenavel_regiao_percentualmlt' if 'ena_armazenavel_regiao_percentualmlt' in df_ena_raw.columns else \
    df_ena_raw.columns[-1]

    df_ena = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ena_raw[col_data_ena], format='mixed'),
        'ENA': limpar_numeros(df_ena_raw[col_val_ena])
    }).drop_duplicates(subset=['Data_Hora']).set_index('Data_Hora')

    # 4. Carga Horária
    col_data_carga = 'din_instante' if 'din_instante' in df_carga_raw.columns else df_carga_raw.columns[0]
    col_val_carga = 'val_cargaenergiahomwmed' if 'val_cargaenergiahomwmed' in df_carga_raw.columns else \
    df_carga_raw.columns[-1]

    df_carga_clean = df_carga_raw.copy()
    df_carga_clean['Data_Hora'] = pd.to_datetime(df_carga_clean[col_data_carga], format='mixed')
    df_carga_clean['CARGA'] = limpar_numeros(df_carga_clean[col_val_carga])
    df_carga = df_carga_clean.groupby('Data_Hora')['CARGA'].mean().to_frame()

    # 5. Merge e Interpolação Temporal
    df_merged = df_pld.join([df_ear, df_ena, df_carga], how='outer').sort_index()
    df_merged['EAR'] = df_merged['EAR'].ffill()
    df_merged['ENA'] = df_merged['ENA'].ffill()
    df_merged.dropna(subset=['PLD', 'CARGA'], inplace=True)

    return df_merged


def buscar_dados_online(anos: range) -> pd.DataFrame:
    """Busca os dados online via URL pública com verificação de requisição."""
    pld_files, ear_files, ena_files, carga_files = [], [], [], []

    for ano in anos:
        print(f"🌐 Baixando dados online do ano {ano}...")

        # PLD (Utiliza arquivo local como fonte do histórico de preços ou requisição)
        caminho_pld_local = RAW_DATA_DIR / f'pld_horario_{ano}.csv'
        if caminho_pld_local.exists():
            pld_files.append(pd.read_csv(caminho_pld_local, sep=';'))
        else:
            raise FileNotFoundError(f"Arquivo base de PLD local ({caminho_pld_local.name}) não encontrado.")

        # ONS - Baixa diretamente da nuvem
        ear_files.append(pd.read_csv(ONS_URLS['EAR'].format(ano=ano), sep=';'))
        ena_files.append(pd.read_csv(ONS_URLS['ENA'].format(ano=ano), sep=';'))
        carga_files.append(pd.read_csv(ONS_URLS['CARGA'].format(ano=ano), sep=';'))

    return processar_e_unificar_dfs(pld_files, ear_files, ena_files, carga_files)


def carregar_dados_locais_fallback(anos: range) -> pd.DataFrame:
    """Função de resgate para carregar os dados persistidos localmente."""
    # 1. Tenta carregar a base consolidada Parquet
    if PARQUET_PATH.exists():
        print("📁 [FALLBACK] Carregando base salva em 'data/processed/pld_historico.parquet'...")
        return pd.read_parquet(PARQUET_PATH)

    # 2. Se não houver Parquet, tenta ler os CSVs brutos locais
    print("📁 [FALLBACK] Parquet não encontrado. Processando a partir dos CSVs locais em 'data/raw'...")
    pld_files = [pd.read_csv(RAW_DATA_DIR / f'pld_horario_{ano}.csv', sep=';') for ano in anos]
    ear_files = [pd.read_csv(RAW_DATA_DIR / f'EAR_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos]
    ena_files = [pd.read_csv(RAW_DATA_DIR / f'ENA_DIARIO_SUBSISTEMA_{ano}.csv', sep=';') for ano in anos]
    carga_files = [pd.read_csv(RAW_DATA_DIR / f'CURVA_CARGA_{ano}.csv', sep=';') for ano in anos]

    return processar_e_unificar_dfs(pld_files, ear_files, ena_files, carga_files)


def carregar_e_unificar_dados(anos: range = range(2021, 2026)) -> pd.DataFrame:
    """
    Estratégia API-First:
    1. Tenta conectar e baixar os dados atualizados das APIs/URLs públicas.
    2. Se tiver sucesso, salva a nova versão em Parquet para atualização do cache.
    3. Em caso de erro de conexão, faz fallback para a base local.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PRIMEIRA TENTATIVA: API / Download Online
    # -------------------------------------------------------------------------
    try:
        print("📡 Conectando aos servidores da ONS/CCEE...")
        df_unificado = buscar_dados_online(anos)

        # Salva a nova versão baixada no Parquet para servir de fallback nas próximas vezes
        try:
            df_unificado.to_parquet(PARQUET_PATH)
            print(f"✅ Dados online obtidos e cache atualizado em: {PARQUET_PATH}")
        except Exception as e_save:
            print(f"⚠️ Não foi possível atualizar o arquivo .parquet: {e_save}")

        return df_unificado

    # -------------------------------------------------------------------------
    # FALLBACK: Se houver qualquer falha de rede/API
    # -------------------------------------------------------------------------
    except Exception as e_online:
        print(f"⚠️ A conexão online com as APIs falhou: {e_online}")
        print("🔄 Ativando modo FALLBACK: Buscando base de dados armazenada localmente...")

        try:
            return carregar_dados_locais_fallback(anos)
        except Exception as e_local:
            raise FileNotFoundError(
                "❌ Erro Crítico: Não foi possível obter dados online (API fora do ar ou sem internet) "
                "nem carregar os dados armazenados localmente."
            ) from e_local