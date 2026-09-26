from pathlib import Path
import pandas as pd

def carregar_dados(caminho, aba) -> pd.DataFrame:
    """Carrega uma aba da planilha Excel com os dados brutos.

    Args:
        caminho: Caminho da planilha Excel.
        aba: Nome da aba que contém o dataset.

    Returns:
        DataFrame com os dados da aba selecionada.
    """
    caminho_raw = Path(caminho)
    df = pd.read_excel(caminho_raw, sheet_name=aba)

    return df

def salvar_parquet(df, caminho) -> None:
    """Salva o DataFrame em Parquet, sem incluir o índice.

    Cria o diretório de destino caso ele não exista.

    Args:
        df: DataFrame a ser salvo.
        caminho: Caminho do arquivo Parquet de saída.
    """
    processed_path = Path(caminho)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(processed_path, index=False)