import pandas as pd
import pytest

from src.data_validation import (
    COLUNAS_MODELO,
    validar_alvo,
    validar_dados,
)


def test_validar_dados_aceita_dataset_valido() -> None:
    dados = {
        coluna: [0, 1]
        for coluna in COLUNAS_MODELO
    }
    dados["Churn Value"] = [0, 1]

    df = pd.DataFrame(dados)

    validar_dados(df, "Churn Value")

def test_validar_dados_rejeita_coluna_obrigatoria_ausente() -> None:
    dados = {
        coluna: [0, 1]
        for coluna in COLUNAS_MODELO
    }
    dados["Churn Value"] = [0, 1]

    df = pd.DataFrame(dados).drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Contract"):
        validar_dados(df, "Churn Value")

@pytest.mark.parametrize(
    "valores",
    [
        [0, 2],
        [0, None],
        ["No", "Yes"],
    ],
)
def test_validar_alvo_rejeita_valores_invalidos(valores: list[object]) -> None:
    df = pd.DataFrame({"Churn Value": valores})

    with pytest.raises(ValueError):
        validar_alvo(df, "Churn Value")