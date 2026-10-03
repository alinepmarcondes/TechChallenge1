"""Configurações compartilhadas pela aplicação de churn."""

import os
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

CAMINHO_MODELO = Path(
    os.getenv(
        "CHURN_MODEL_PATH",
        RAIZ_PROJETO / "models" / "logistic_regression_final.joblib",
    )
)

NOME_MODELO = "LogisticRegression"
VERSAO_MODELO = "1.0.0"