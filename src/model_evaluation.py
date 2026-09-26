import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def avaliar_modelo(
    pipeline: Pipeline,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> dict[str, float]:
    """Avalia o modelo no conjunto de teste.

    Calcula accuracy, precision, recall e F1-score para a classe
    positiva (churn = 1), além de ROC-AUC com as probabilidades
    previstas para essa classe.

    Args:
        pipeline: Pipeline treinado.
        X_teste: Variáveis de entrada do conjunto de teste.
        y_teste: Valores reais de churn do conjunto de teste.

    Returns:
        Dicionário com o nome e o valor de cada métrica.
    """
    y_pred = pipeline.predict(X_teste)
    y_pred_proba = pipeline.predict_proba(X_teste)[:, 1]

    return {
        "accuracy": accuracy_score(y_teste, y_pred),
        "precision": precision_score(y_teste, y_pred, zero_division=0),
        "recall": recall_score(y_teste, y_pred, zero_division=0),
        "f1_score": f1_score(y_teste, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_teste, y_pred_proba),
    }


def salvar_metricas(
    metricas: dict[str, float],
    caminho: str | Path,
) -> None:
    """Salva as métricas de avaliação em um arquivo JSON.

    Cria o diretório de destino caso ele não exista.

    Args:
        metricas: Dicionário produzido por `avaliar_modelo`.
        caminho: Caminho do arquivo JSON de saída.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, indent=2)