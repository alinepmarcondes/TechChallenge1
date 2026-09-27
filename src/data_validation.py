import pandas as pd

COLUNAS_MODELO = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

def validar_colunas(df: pd.DataFrame, colunas_obrigatorias: list[str]) -> None:
    """Verifica se todas as colunas obrigatórias estão no dataset.

    Args:
        df: DataFrame a ser validado.
        colunas_obrigatorias: Nomes das colunas esperadas.

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente.
    """
    ausentes = set(colunas_obrigatorias) - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")


def validar_alvo(df: pd.DataFrame, coluna_alvo: str) -> None:
    """Valida a coluna utilizada como alvo da classificacao de churn.

    Confirma que a coluna existe, não contem valores ausentes, pode ser
    convertida para numeros inteiros e possui somente as classes binárias
    0 (nao churn) e 1 (churn). A função apenas valida os dados e não altera
    o DataFrame recebido.

    Args:
        df: DataFrame que contem a coluna alvo.
        coluna_alvo: Nome da coluna utilizada como target do modelo.

    Raises:
        ValueError: Se a coluna alvo estiver ausente, contiver valores
            ausentes, nao puder ser convertida para inteiro ou apresentar
            valores diferentes de 0 e 1.
    """
    validar_colunas(df, [coluna_alvo])

    if df[coluna_alvo].isna().any():
        raise ValueError(f"'{coluna_alvo}' contém valores ausentes.")

    try:
        alvo = df[coluna_alvo].astype(int)
    except (TypeError, ValueError) as erro:
        raise ValueError(
            f"'{coluna_alvo}' não pôde ser convertida para inteiro."
        ) from erro

    valores = set(alvo.unique())

    if not valores <= {0, 1}:
        raise ValueError(
            f"'{coluna_alvo}' deve conter apenas 0 e 1."
        )


def validar_dados(df: pd.DataFrame, coluna_alvo: str) -> None:
    """Executa as validações iniciais do dataset.

    Verifica se o DataFrame não está vazio, se a coluna
    `Total Charges` existe e se a coluna alvo é válida.

    Args:
        df: DataFrame carregado da planilha.
        coluna_alvo: Nome da coluna que indica churn.

    Raises:
        ValueError: Se alguma das validações falhar.
    """
    if df.empty:
        raise ValueError("O dataset está vazio.")

    validar_colunas(
        df,
        [coluna_alvo, *COLUNAS_MODELO],
    )
    validar_alvo(df, coluna_alvo)