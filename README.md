# Previsão do PLD Horário com XGBoost (Mercado de Energia Elétrica B2B)

Modelo de Machine Learning para previsão do Preço de Liquidação das Diferenças (PLD) horário no Subsistema Sudeste/Centro-Oeste do Brasil.

## 📌 Arquitetura e Engenharia
- **Dados:** Integração de séries temporais de Carga, ENA (Energia Natural Afluente) e EAR (Energia Armazenada) obtidas via ONS e CCEE.
- **Engenharia de Features:** Lags temporais, médias móveis e sazonalidade de calendário.
- **Validação Cruzada Cega:** Separação estritamente cronológica (80% Treino / 10% Validação / 10% Teste) sem contaminação temporal (*data leakage*).
- **Regras de Negócio:** Aplicação do Piso Regulatório de PLD real.

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone [https://github.com/seu-usuario/pld-forecasting-xgboost.git](https://github.com/seu-usuario/pld-forecasting-xgboost.git)
cd pld-forecasting-xgboost