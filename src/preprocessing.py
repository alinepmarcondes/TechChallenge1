import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUNAS_DESCARTADAS = [
    "CustomerID",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]


def converter_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `Total Charges` para valores numéricos.

    Valores que não puderem ser convertidos serão representados como
    ausentes (`NaN`) e tratados posteriormente pelo pipeline.

    Args:
        df: DataFrame que contém a coluna `Total Charges`.

    Returns:
        Cópia do DataFrame com `Total Charges` em formato numérico.
    """
    resultado = df.copy()
    resultado["Total Charges"] = pd.to_numeric(
        resultado["Total Charges"],
        errors="coerce",
    )
    return resultado


def separar_variaveis_alvo(
    df: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as variáveis de entrada da coluna alvo.

    Remove das variáveis de entrada as colunas identificadoras,
    redundantes e associadas a vazamento do alvo, conforme a lista
    `COLUNAS_DESCARTADAS`. Converte o alvo para inteiro.

    Args:
        df: DataFrame após a conversão de `Total Charges`.
        coluna_alvo: Nome da coluna que indica churn.

    Returns:
        Tupla contendo o DataFrame de variáveis de entrada (`X`)
        e a série com o alvo (`y`).
    """
    colunas_para_descartar = [
        coluna for coluna in [coluna_alvo, *COLUNAS_DESCARTADAS]
        if coluna in df.columns
    ]

    X = df.drop(columns=colunas_para_descartar)
    y = df[coluna_alvo].astype(int)

    return X, y


def criar_preprocessador(X_treino: pd.DataFrame) -> ColumnTransformer:
    """Monta o pré-processamento.

    Identifica colunas numéricas e categóricas a partir do treino.
    Para as numéricas, aplica imputação pela mediana e padronização.
    Para as categóricas, aplica imputação pela moda e one-hot encoding.

    O objeto retornado ainda não é ajustado aos dados. Seu ajuste
    acontece durante o treinamento do pipeline completo.

    Args:
        X_treino: Variáveis de entrada do conjunto de treino.

    Returns:
        Transformador que prepara as variáveis para o modelo.
    """
    colunas_numericas = X_treino.select_dtypes(
        include=["number"]
    ).columns.tolist()

    colunas_categoricas = X_treino.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="if_binary",
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numerico, colunas_numericas),
        ("categorical", categorico, colunas_categoricas),
    ])