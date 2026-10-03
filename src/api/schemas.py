
"""Schemas de entrada e saida da API de churn."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

SimNao = Literal["Yes", "No"]
Genero = Literal["Female", "Male"]
ServicoInternet = Literal["DSL", "Fiber optic", "No"]
ServicoOpcional = Literal["Yes", "No", "No internet service"]
MultiplasLinhas = Literal["Yes", "No", "No phone service"]
Contrato = Literal["Month-to-month", "One year", "Two year"]
MetodoPagamento = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

class ClienteInput(BaseModel):
    """Dados de um cliente necessarios para a previsao de churn."""

    model_config = ConfigDict(extra="forbid")

    gender: Genero
    senior_citizen: SimNao
    partner: SimNao
    dependents: SimNao
    tenure_months: int = Field(ge=0)
    phone_service: SimNao
    multiple_lines: MultiplasLinhas
    internet_service: ServicoInternet
    online_security: ServicoOpcional
    online_backup: ServicoOpcional
    device_protection: ServicoOpcional
    tech_support: ServicoOpcional
    streaming_tv: ServicoOpcional
    streaming_movies: ServicoOpcional
    contract: Contrato
    paperless_billing: SimNao
    payment_method: MetodoPagamento
    monthly_charges: float = Field(ge=0)
    total_charges: float | None = Field(default=None, ge=0)
    cltv: int = Field(ge=0)

class PredicaoOutput(BaseModel):
    """Resultado retornado pelo endpoint de predicao."""

    previsao_churn: Literal[0, 1]
    probabilidade_churn: float = Field(ge=0, le=1)

class HealthOutput(BaseModel):
    """Estado atual da API e do carregamento do modelo."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str