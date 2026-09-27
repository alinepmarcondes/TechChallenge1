import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model_predictor import carregar_modelo, prever_churn


def criar_pipeline_treinada() -> tuple[Pipeline, pd.DataFrame]:
    X = pd.DataFrame(
        {
            "Tenure Months": [1, 2, 24, 36, 48, 60],
            "Monthly Charges": [95.0, 90.0, 75.0, 55.0, 45.0, 35.0],
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0])

    pipeline = Pipeline(
        [("classifier", LogisticRegression(random_state=42))]
    )
    pipeline.fit(X, y)
    return pipeline, X


def test_carregar_modelo_preserva_predicoes(tmp_path) -> None:
    pipeline, X = criar_pipeline_treinada()
    caminho = tmp_path / "modelo.joblib"
    joblib.dump(pipeline, caminho)

    modelo_carregado = carregar_modelo(caminho)

    np.testing.assert_array_equal(
        pipeline.predict(X),
        modelo_carregado.predict(X),
    )


def test_prever_churn_retorna_classe_e_probabilidade() -> None:
    pipeline, X = criar_pipeline_treinada()

    resultado = prever_churn(pipeline, X.iloc[:2])

    assert list(resultado.columns) == [
        "previsao_churn",
        "probabilidade_churn",
    ]
    assert len(resultado) == 2
    assert resultado["previsao_churn"].isin([0, 1]).all()
    assert resultado["probabilidade_churn"].between(0, 1).all()


def test_prever_churn_rejeita_dataframe_vazio() -> None:
    pipeline, X = criar_pipeline_treinada()

    with pytest.raises(ValueError, match="Não há dados"):
        prever_churn(pipeline, X.iloc[0:0])


def test_carregar_modelo_inexistente_falha(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        carregar_modelo(tmp_path / "inexistente.joblib")