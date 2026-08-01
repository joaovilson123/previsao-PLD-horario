import requests
import pandas as pd

# Endpoint oficial da API do ONS
ONS_API_URL = "https://dados.ons.org.br/api/3/action/datastore_search"

# IDs dos recursos na Plataforma de Dados Abertos
RESOURCE_IDS = {
    'EAR': 'b155239a-5847-493e-a837-775cf015f206',  # EAR Diário por Subsistema
    'ENA': '21a711ef-7529-43c2-bf72-6ef04a9d0739',  # ENA Diário por Subsistema
    'CARGA': '1b33230a-3c13-40e1-b1e9-4e00508544d6'  # Curva de Carga Horária
}


def buscar_dados_ons_api(tipo_dado: str, limit: int = 10000) -> pd.DataFrame:
    """
    Consulta a API do ONS para buscar registros recentes de EAR, ENA ou CARGA.

    Parameters:
        tipo_dado (str): 'EAR', 'ENA' ou 'CARGA'
        limit (int): Quantidade de registros recentes a retornar
    """
    if tipo_dado not in RESOURCE_IDS:
        raise ValueError(f"Tipo de dado inválido. Escolha entre: {list(RESOURCE_IDS.keys())}")

    resource_id = RESOURCE_IDS[tipo_dado]

    # Parâmetros da requisição HTTP
    params = {
        'resource_id': resource_id,
        'limit': limit
    }

    try:
        response = requests.get(ONS_API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Converte a resposta JSON em DataFrame
        records = data['result']['records']
        df = pd.DataFrame(records)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar na API do ONS para {tipo_dado}: {e}")
        return pd.DataFrame()


def buscar_dados_ons_direto_url(ano: int, tipo_dado: str) -> pd.DataFrame:
    """
    Lê o dataset completo de um ano específico direto da URL pública do ONS sem baixar manualmente.
    """
    urls = {
        'CARGA': f'https://dados.ons.org.br/dataset/curva-carga-2/resource/CURVA_CARGA_{ano}.csv',
        'EAR': f'https://dados.ons.org.br/dataset/ear-diario-subsistema/resource/EAR_DIARIO_SUBSISTEMA_{ano}.csv',
        'ENA': f'https://dados.ons.org.br/dataset/ena-diario-subsistema/resource/ENA_DIARIO_SUBSISTEMA_{ano}.csv'
    }

    url = urls.get(tipo_dado)
    if not url:
        raise ValueError("Tipo de dado não encontrado.")

    print(f"Baixando {tipo_dado} ({ano}) direto do ONS...")
    # O Pandas baixa o arquivo direto da URL em memória
    df = pd.read_csv(url, sep=';')
    return df