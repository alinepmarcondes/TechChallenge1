from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def criar_pipeline_modelo(
    preprocessador: ColumnTransformer,
    random_state: int = 42,
) -> Pipeline:
    """Cria o pipeline da regressão logística baseline.

    Combina o pré-processador com uma regressão logística configurada
    com `max_iter=1000`, conforme o notebook.

    Args:
        preprocessador: Transformador das variáveis de entrada.
        random_state: Seed usada pela regressão logística.

    Returns:
        Pipeline ainda não treinado.
    """
    return Pipeline([
        ("preprocessor", preprocessador),
        (
            "classifier",
            LogisticRegression(
                max_iter=1_000,
                random_state=random_state,
            ),
        ),
    ])


def treinar_modelo(
    pipeline: Pipeline,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
) -> Pipeline:
    """Ajusta o pré-processamento e o modelo com os dados de treino.

    Args:
        pipeline: Pipeline ainda não treinado.
        X_treino: Variáveis de entrada do conjunto de treino.
        y_treino: Valores reais de churn do conjunto de treino.

    Returns:
        O próprio pipeline após o ajuste.
    """
    pipeline.fit(X_treino, y_treino)
    return pipeline


def salvar_modelo(pipeline: Pipeline, caminho: str | Path) -> None:
    """Salva o pipeline completo em um arquivo Joblib.

    Cria o diretório de destino caso ele não exista. O pipeline salvo
    inclui o pré-processamento ajustado e o classificador treinado.

    Args:
        pipeline: Pipeline treinado.
        caminho: Caminho do arquivo `.joblib` de saída.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho)