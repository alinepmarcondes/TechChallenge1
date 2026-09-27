import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.preprocessing import converter_total_charges, separar_variaveis_alvo


def test_converter_total_charges_sem_alterar_dataframe_original() -> None:
    df = pd.DataFrame({
        "Total Charges": ["100.50", " ", "invalido"],
    })
    original = df.copy(deep=True)

    resultado = converter_total_charges(df)

    pd.testing.assert_frame_equal(df, original)

    assert is_numeric_dtype(resultado["Total Charges"])
    assert resultado.loc[0, "Total Charges"] == 100.50
    assert pd.isna(resultado.loc[1, "Total Charges"])
    assert pd.isna(resultado.loc[2, "Total Charges"])


def test_separar_variaveis_remove_alvo_e_colunas_de_vazamento() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": ["A", "B"],
            "Churn Label": ["No", "Yes"],
            "Churn Score": [10, 90],
            "Churn Reason": [None, "Competitor"],
            "Churn Value": [0, 1],
            "Tenure Months": [12, 2],
            "Monthly Charges": [50.0, 90.0],
        }
    )

    X, y = separar_variaveis_alvo(df, "Churn Value")

    assert list(X.columns) == ["Tenure Months", "Monthly Charges"]
    assert y.tolist() == [0, 1]
    assert y.dtype.kind in {"i", "u"}