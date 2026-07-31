from pathlib import Path
import os
import pandas as pd

# Define diretórios base a partir da localização do script
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PARQUET_PATH = PROCESSED_DATA_DIR / "pld_historico.parquet"

# Mapeamento das URLs oficiais do ONS e CCEE para fallback online
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


def ler_base_ou_baixar(tipo: str, ano: int, raw_dir: Path) -> pd.DataFrame:
    """
    Busca o CSV localmente em data/raw. Se não encontrar, faz o download
    direto da URL oficial do ONS/CCEE.
    """
    # Nomes padrões dos arquivos locais
    nomes_locais = {
        'PLD': f'pld_horario_{ano}.csv',
        'EAR': f'EAR_DIARIO_SUBSISTEMA_{ano}.csv',
        'ENA': f'ENA_DIARIO_SUBSISTEMA_{ano}.csv',
        'CARGA': f'CURVA_CARGA_{ano}.csv'
    }

    caminho_local = raw_dir / nomes_locais[tipo]

    # 1. Tenta carregar do disco local
    if caminho_local.exists():
        return pd.read_csv(caminho_local, sep=';')

    # 2. Se não existir no disco, carrega direto da URL
    print(f"⚠️  Arquivo local {nomes_locais[tipo]} não encontrado. Baixando da nuvem...")
    if tipo in ONS_URLS:
        url = ONS_URLS[tipo].format(ano=ano)
        return pd.read_csv(url, sep=';')

    raise FileNotFoundError(f"Não foi possível localizar localmente nem online a base {tipo} para o ano {ano}.")


def carregar_e_unificar_dados(anos: range = range(2021, 2026), force_reprocess: bool = False) -> pd.DataFrame:
    """
    Carrega, trata e unifica os dados do mercado de energia.

    Verifica se já existe a base consolidada em formato Parquet para ganho de performance.
    Caso contrário, processa das fontes locais/online e gera o Parquet.
    """
    # Garantir que as pastas de dados existam
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. RETORNO RÁPIDO: Se o parquet existir, carrega em menos de 1 segundo
    if PARQUET_PATH.exists() and not force_reprocess:
        print("⚡ Carregando base unificada a partir de 'data/processed/pld_historico.parquet'...")
        return pd.read_parquet(PARQUET_PATH)

    print("🔄 Processando arquivos brutos (Locais / ONS / CCEE)...")

    # 2. CARREGAMENTO DOS CSVs (Locais ou via URL)
    pld_files = [ler_base_ou_baixar('PLD', ano, RAW_DATA_DIR) for ano in anos]
    ear_files = [ler_base_ou_baixar('EAR', ano, RAW_DATA_DIR) for ano in anos]
    ena_files = [ler_base_ou_baixar('ENA', ano, RAW_DATA_DIR) for ano in anos]
    carga_files = [ler_base_ou_baixar('CARGA', ano, RAW_DATA_DIR) for ano in anos]

    df_pld_raw = pd.concat(pld_files, ignore_index=True)
    df_ear_raw = pd.concat(ear_files, ignore_index=True)
    df_ena_raw = pd.concat(ena_files, ignore_index=True)
    df_carga_raw = pd.concat(carga_files, ignore_index=True)

    # 3. TRATAMENTO E FORMATO DE DATAS (Suporta variações de nomenclatura de colunas)

    # PLD Horário
    df_pld = pd.DataFrame({
        'Data_Hora': pd.to_datetime({
            'year': df_pld_raw['MES_REFERENCIA'] // 100,
            'month': df_pld_raw['MES_REFERENCIA'] % 100,
            'day': df_pld_raw['DIA'],
            'hour': df_pld_raw['HORA']
        }),
        'PLD': limpar_numeros(df_pld_raw['PLD_HORA'])
    }).set_index('Data_Hora')

    # EAR Diário
    col_data_ear = 'ear_data' if 'ear_data' in df_ear_raw.columns else df_ear_raw.columns[0]
    col_val_ear = 'ear_verif_subsistema_percentual' if 'ear_verif_subsistema_percentual' in df_ear_raw.columns else \
    df_ear_raw.columns[-1]

    df_ear = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ear_raw[col_data_ear], format='mixed'),
        'EAR': limpar_numeros(df_ear_raw[col_val_ear])
    }).drop_duplicates(subset=['Data_Hora']).set_index('Data_Hora')

    # ENA Diário
    col_data_ena = 'ena_data' if 'ena_data' in df_ena_raw.columns else df_ena_raw.columns[0]
    col_val_ena = 'ena_armazenavel_regiao_percentualmlt' if 'ena_armazenavel_regiao_percentualmlt' in df_ena_raw.columns else \
    df_ena_raw.columns[-1]

    df_ena = pd.DataFrame({
        'Data_Hora': pd.to_datetime(df_ena_raw[col_data_ena], format='mixed'),
        'ENA': limpar_numeros(df_ena_raw[col_val_ena])
    }).drop_duplicates(subset=['Data_Hora']).set_index('Data_Hora')

    # Carga Horária
    col_data_carga = 'din_instante' if 'din_instante' in df_carga_raw.columns else df_carga_raw.columns[0]
    col_val_carga = 'val_cargaenergiahomwmed' if 'val_cargaenergiahomwmed' in df_carga_raw.columns else \
    df_carga_raw.columns[-1]

    # Agrupa por horário caso o arquivo bruto contenha múltiplos subsistemas
    df_carga_clean = df_carga_raw.copy()
    df_carga_clean['Data_Hora'] = pd.to_datetime(df_carga_clean[col_data_carga], format='mixed')
    df_carga_clean['CARGA'] = limpar_numeros(df_carga_clean[col_val_carga])
    df_carga = df_carga_clean.groupby('Data_Hora')['CARGA'].mean().to_frame()

    # 4. JUNÇÃO E INTERPOLAÇÃO TEMPORAL
    df_merged = df_pld.join([df_ear, df_ena, df_carga], how='outer').sort_index()

    # Preenche os dados diários (EAR/ENA) para todas as horas do dia
    df_merged['EAR'] = df_merged['EAR'].ffill()
    df_merged['ENA'] = df_merged['ENA'].ffill()

    # Remove registros sem alvo ou sem carga
    df_merged.dropna(subset=['PLD', 'CARGA'], inplace=True)

    # 5. SALVA O ARQUIVO COMPACTADO PARQUET
    try:
        df_merged.to_parquet(PARQUET_PATH)
        print(f"✅ Base unificada gerada com sucesso e salva em: {PARQUET_PATH}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar o arquivo .parquet: {e}")

    return df_merged