import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import carregar_e_unificar_dados
from src.features import criar_features_temporais
from src.model import preparar_dados_e_treinar

# Configuração da página Web
st.set_page_config(
    page_title="Dashboard Preditivo de PLD",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------------------------------------------
# OTIMIZAÇÃO: Caching para a aplicação carregar rápido
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def carregar_dados_processados():
    df_raw = carregar_e_unificar_dados()
    df_feat = criar_features_temporais(df_raw)
    return df_feat

@st.cache_resource
def treinar_modelo_cache(df_feat):
    return preparar_dados_e_treinar(df_feat)

# -----------------------------------------------------------------------------
# INTERFACE GRÁFICA (UI)
# -----------------------------------------------------------------------------
st.title("⚡ Previsão de PLD Horário - Mercado de Energia")
st.markdown("Modelo de Machine Learning (XGBoost) para previsão do Preço de Liquidação das Diferenças.")

# Barra Lateral (Sidebar)
st.sidebar.header("⚙️ Configurações do Painel")
janela_horas = st.sidebar.slider("Janela de Visualização (Horas do Teste):", min_value=24, max_value=1000, value=168, step=24)

# Carregando dados com efeito visual de "Spinner"
with st.spinner("Carregando bases da CCEE/ONS e treinando modelo..."):
    df_feat = carregar_dados_processados()
    model, X_test, y_test, y_pred = treinar_modelo_cache(df_feat)

# CÁLCULO DE MÉTRICAS RÁPIDAS
mae = float(np.mean(np.abs(y_test - y_pred)))
rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
pld_medio_previsto = float(y_pred[-janela_horas:].mean())

# Painel de Métricas em Destaque (Cartões KPI)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Erro Médio (MAE)", f"R$ {mae:.2f}/MWh")
col2.metric("Erro Quadrático (RMSE)", f"R$ {rmse:.2f}/MWh")
col3.metric("PLD Médio Previsto (Período)", f"R$ {pld_medio_previsto:.2f}/MWh")
col4.metric("Status do Modelo", "Ativo", delta="XGBoost v1.0")

st.divider()

# -----------------------------------------------------------------------------
# GRÁFICO INTERATIVO DE ADERÊNCIA TEMPORAL (PLOTLY)
# -----------------------------------------------------------------------------
st.subheader("📊 Aderência Temporal: PLD Real vs. Previsto")

# Seleção do recorte de exibição baseado no Slider da Sidebar
datas_corte = y_test.index[-janela_horas:]
y_test_corte = y_test.iloc[-janela_horas:]
y_pred_corte = y_pred[-janela_horas:]

fig = go.Figure()

# Linha do PLD Real
fig.add_trace(go.Scatter(
    x=datas_corte,
    y=y_test_corte,
    mode='lines',
    name='PLD Real',
    line=dict(color='#2ca02c', width=2)
))

# Linha do PLD Previsto
fig.add_trace(go.Scatter(
    x=datas_corte,
    y=y_pred_corte,
    mode='lines',
    name='PLD Previsto (XGBoost)',
    line=dict(color='#d62728', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title="Data / Hora",
    yaxis_title="Preço (R$/MWh)",
    hovermode="x unified",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Tabela com os dados brutos de previsão recentes
with st.expander("🔍 Ver Tabela de Dados Recentes"):
    df_resultado = pd.DataFrame({
        'Data/Hora': datas_corte,
        'PLD Real (R$)': y_test_corte.values,
        'PLD Previsto (R$)': y_pred_corte
    }).set_index('Data/Hora')
    st.dataframe(df_resultado.style.format("{:.2f}"))