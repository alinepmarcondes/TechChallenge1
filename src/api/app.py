"""Aplicacao FastAPI para inferencia do modelo de churn."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from sklearn.pipeline import Pipeline

from src.api.model_service import realizar_predicao
from src.api.schemas import ClienteInput, HealthOutput, PredicaoOutput
from src.config import CAMINHO_MODELO, NOME_MODELO, VERSAO_MODELO
from src.model_predictor import carregar_modelo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Carrega o modelo uma vez durante a inicializacao da API."""
    try: 
        app.state.modelo = carregar_modelo(CAMINHO_MODELO)
        app.state.erro_modelo = None
    except(FileNotFoundError, TypeError, OSError) as erro:
        app.state.modelo = None
        app.state.erro_modelo= str(erro)

    yield
    
    app.state.modelo = None

app = FastAPI(
    title="API de Previsão de Churn",
    description="API para estimar a propensão de cancelamento de clientes de uma operadora de telecomunicações",
    version="1.0.0",
    lifespan=lifespan
)

@app.get(
    "/health",
    response_model=HealthOutput,
    tags=["Health"],
)
def health(request: Request, response: Response) -> HealthOutput:
    """Informa se a API iniciou e se o modelo esta disponivel."""
    modelo_carregado = request.app.state.modelo is not None

    if not modelo_carregado:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthOutput(
        status="ok" if modelo_carregado else "degraded",
        model_loaded=modelo_carregado,
        model_name=NOME_MODELO,
        model_version=VERSAO_MODELO,
    )

@app.post(
    "/predict",
    response_model=PredicaoOutput,
    tags=["Prediction"],
)
def predict(request: Request, cliente: ClienteInput) -> PredicaoOutput:
    """Retorna a classe e a probabilidade estimada de churn."""
    modelo: Pipeline | None = request.app.state.modelo

    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível para inferência",
        )

    return realizar_predicao(modelo, cliente)