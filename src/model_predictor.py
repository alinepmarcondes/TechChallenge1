from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def carregar_modelo(caminho: str | Path) -> Pipeline:
    """Carrega do disco a pipeline treinada usada para inferência.

    Args:
        caminho: Caminho do arquivo Joblib que contem a pipeline completa.

    Returns:
        Pipeline treinada, incluindo pre-processamento e classificador.

    Raises:
        FileNotFoundError: Se o arquivo informado não existir.
        TypeError: Se o objeto carregado nao for uma Pipeline do Scikit-Learn.
    """
    caminho = Path(caminho)

    if not caminho.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")

    modelo = joblib.load(caminho)

    if not isinstance(modelo, Pipeline):
        raise TypeError("O arquivo carregado não contem uma Pipeline válida.")

    return modelo


def prever_churn(
    modelo: Pipeline,
    dados: pd.DataFrame,
) -> pd.DataFrame:
    """Gera a classe e a probabilidade de churn para novos clientes.

    O DataFrame de entrada deve conter as mesmas features utilizadas no
    treinamento. O pre-processamento e aplicado pela propria pipeline.

    Args:
        modelo: Pipeline treinada contendo pre-processamento e classificador.
        dados: Dados dos clientes que serão avaliados.

    Returns:
        DataFrame com ``previsao_churn`` (0 ou 1) e
        ``probabilidade_churn`` (valor entre 0 e 1), preservando o índice
        da entrada.

    Raises:
        ValueError: Se o DataFrame de entrada estiver vazio.
    """
    if dados.empty:
        raise ValueError("Não há dados para realizar a predição.")

    previsoes = modelo.predict(dados)
    probabilidades = modelo.predict_proba(dados)[:, 1]

    return pd.DataFrame(
        {
            "previsao_churn": previsoes.astype(int),
            "probabilidade_churn": probabilidades,
        },
        index=dados.index,
    )