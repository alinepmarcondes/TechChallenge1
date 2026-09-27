from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_ingestion import carregar_dados, salvar_parquet
from src.data_validation import validar_dados
from src.model_evaluation import avaliar_modelo, salvar_metricas
from src.model_trainer import (
    criar_pipeline_modelo,
    salvar_modelo,
    treinar_modelo,
)
from src.preprocessing import (
    converter_total_charges,
    criar_preprocessador,
    separar_variaveis_alvo,
)

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET = "Churn Value"


def main() -> None:
    """Executa o fluxo de treinamento e avaliação do modelo.

    Carrega e valida os dados, converte `Total Charges`, salva uma
    cópia em Parquet, separa treino e teste, treina o pipeline e
    salva o modelo e suas métricas.
    """
    df = carregar_dados(
        RAIZ_PROJETO / "data/raw/Telco_customer_churn.xlsx",
        aba="Telco_Churn",
    )

    validar_dados(df, TARGET)
    df = converter_total_charges(df)
    salvar_parquet(df, RAIZ_PROJETO / "data/processed/dataset.parquet")

    X, y = separar_variaveis_alvo(df, TARGET)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessador = criar_preprocessador(X_treino)
    pipeline = criar_pipeline_modelo(preprocessador, RANDOM_STATE)
    pipeline = treinar_modelo(pipeline, X_treino, y_treino)

    metricas = avaliar_modelo(pipeline, X_teste, y_teste)

    salvar_modelo(pipeline, RAIZ_PROJETO / "models/logistic_regression_final.joblib")
    resultado = {
        "modelo": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "max_iter": 1_000,
        "target": TARGET,
        "metricas": metricas,
    }
    salvar_metricas(
        resultado,
        RAIZ_PROJETO / "reports/logistic_regression_final_metrics.json",
    )

    print(resultado)


if __name__ == "__main__":
    main()