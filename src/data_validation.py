import pandas as pd


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
    """Verifica a presença e os valores da coluna alvo.

    A coluna deve existir, não conter valores ausentes e possuir apenas
    as classes 0 e 1 após a conversão para inteiro.

    Args:
        df: DataFrame a ser validado.
        coluna_alvo: Nome da coluna que indica churn.

    Raises:
        ValueError: Se a coluna estiver ausente, contiver valores
            ausentes ou apresentar classes diferentes de 0 e 1.
        ValueError: Se algum valor não puder ser convertido para inteiro.
    """
    validar_colunas(df, [coluna_alvo])

    if df[coluna_alvo].isna().any():
        raise ValueError(f"'{coluna_alvo}' contém valores ausentes.")

    valores = set(df[coluna_alvo].astype(int).unique())
    if not valores <= {0, 1}:
        raise ValueError(f"'{coluna_alvo}' deve conter apenas 0 e 1.")


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

    validar_colunas(df, ["Total Charges"])
    validar_alvo(df, coluna_alvo)