"""Adaptação entre os schemas HTTP e a pipeline do Scikit-Learn."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.api.schemas import ClienteInput, PredicaoOutput
from src.model_predictor import prever_churn

MAPEAMENTO_COLUNAS = {
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure_months": "Tenure Months",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "cltv": "CLTV",
}

def cliente_para_dataframe(cliente: ClienteInput) -> pd.DataFrame:
    """Converte o schema recebido pela API para o formato do modelo.

    Args:
        cliente: Dados validados de um cliente.

    Returns:
        DataFrame com uma linha e nomes de colunas iguais aos utilizados
        durante o treinamento.
    """
    dados = cliente.model_dump()
    dados_modelo = {
        MAPEAMENTO_COLUNAS[nome]: valor 
        for nome, valor in dados.items()
    }
    return pd.DataFrame([dados_modelo])

def realizar_predicao(
    modelo: Pipeline,
    cliente: ClienteInput,
) -> PredicaoOutput:
    """Executa a inferencia e formata a resposta da API.

    Args:
        modelo: Pipeline completa e previamente treinada.
        cliente: Dados validados do cliente.

    Returns:
        Classe prevista e probabilidade estimada de churn.
    """
    dados = cliente_para_dataframe(cliente)
    resultado = prever_churn(modelo, dados).iloc[0]

    return PredicaoOutput(
        previsao_churn=int(resultado["previsao_churn"]),
        probabilidade_churn=float(resultado["probabilidade_churn"])
    )