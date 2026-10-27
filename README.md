[![Vídeo no YouTube](https://img.shields.io/badge/YouTube-Apresentação-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/pLkH5l2uMoo)

# Tech Challenge - Previsão de Churn

Pipeline de Machine Learning para identificar clientes de telecomunicações com maior propensão ao churn. O projeto cobre análise exploratória, comparação de modelos, seleção e persistência do modelo campeão, modularização do código, testes automatizados e disponibilização de inferência por uma API REST com FastAPI.

Projeto desenvolvido na Fase 1 da pós-graduação em Machine Learning Engineering da FIAP.

## Visão geral

Uma operadora de telecomunicações precisa identificar clientes com risco de cancelamento para apoiar ações preventivas de retenção. A solução recebe dados cadastrais, contratuais e de consumo de um cliente e retorna:

- a classe prevista: permanência (`0`) ou churn (`1`);
- a probabilidade estimada de churn.

A Regressão Logística foi escolhida como modelo final após comparação com Random Forest e MLPClassifier. Além do melhor F1-score médio na validação cruzada, ela apresentou boa capacidade de generalização, menor custo computacional e maior interpretabilidade.

## Objetivos

- explorar e validar os dados de clientes;
- preparar variáveis numéricas e categóricas sem vazamento de dados;
- treinar Regressão Logística, Random Forest e MLPClassifier;
- avaliar a robustez dos modelos por validação cruzada estratificada;
- selecionar e salvar o modelo campeão;
- organizar treinamento e inferência em módulos Python;
- expor o modelo por meio de uma API FastAPI;
- garantir o funcionamento das partes principais com testes automatizados;
- permitir a execução local e em container Docker.

## Dataset

Foi utilizado o **Telco Customer Churn**, conjunto público da IBM com mais de 7 mil registros de clientes e atributos relacionados a perfil, serviços contratados, permanência, cobranças e cancelamento.

O alvo do problema é `Churn Value`:

- `0`: cliente permaneceu;
- `1`: cliente cancelou o serviço.

Antes do treinamento, são removidas colunas identificadoras, constantes, geográficas redundantes e com vazamento direto ou indireto do alvo, como `CustomerID`, `Churn Label`, `Churn Score` e `Churn Reason`.

> Os dados não devem ser utilizados para identificar pessoas ou tomar decisões automatizadas sem supervisão humana. Consulte também o [Model Card](docs/MODEL_CARD.md).

### Premissa sobre o uso de `CLTV`

Para fins acadêmicos, assume-se que a variável `CLTV` está disponível no
momento da inferência e é calculada pelo mesmo processo utilizado na construção
do dataset. Em uma aplicação real, essa premissa deve ser validada antes da
implantação. Caso a variável não esteja disponível ou seja calculada de forma
diferente, o modelo deverá ser retreinado sem `CLTV`.

## Metodologia

### Pré-processamento

O pré-processamento faz parte da própria `Pipeline` do Scikit-Learn, evitando que informações do conjunto de teste sejam usadas durante o ajuste:

- conversão de `Total Charges` para formato numérico;
- imputação de valores numéricos ausentes pela mediana;
- padronização das variáveis numéricas com `StandardScaler`;
- imputação de valores categóricos ausentes pela moda;
- codificação das categorias com `OneHotEncoder`;
- tratamento de categorias desconhecidas durante a inferência;
- separação estratificada entre treino e teste.

### Modelos avaliados

- Regressão Logística;
- Random Forest;
- rede neural simples com `MLPClassifier`.

Os três modelos foram avaliados com validação cruzada estratificada em cinco folds. As métricas técnicas consideradas foram accuracy, precision, recall, F1-score e ROC-AUC. O F1-score foi utilizado como principal referência de seleção por equilibrar precision e recall em um problema com classes desbalanceadas.

Como métrica de negócio, recomenda-se acompanhar a quantidade ou proporção de clientes com churn corretamente identificados e o retorno das campanhas de retenção direcionadas pelo modelo. Essa avaliação depende dos custos e benefícios reais da operação e não é mensurada diretamente pelo dataset.

## Resultados

### Validação cruzada

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Regressão Logística** | **0.8113** | 0.6656 | **0.5806** | **0.6196** | **0.8588** |
| Random Forest | 0.7993 | 0.6521 | 0.5244 | 0.5811 | 0.8446 |
| MLPClassifier | 0.8037 | **0.6708** | 0.5130 | 0.5810 | 0.8512 |

O Random Forest atingiu métricas perfeitas no treino, mas apresentou queda relevante na validação, indicando overfitting na configuração avaliada. A Regressão Logística apresentou a melhor combinação entre desempenho de validação, estabilidade e simplicidade.

### Modelo campeão no conjunto de teste

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8034 |
| Precision | 0.6456 |
| Recall | 0.5749 |
| F1-score | 0.6082 |
| ROC-AUC | 0.8488 |

O artefato final contém o pré-processamento ajustado e a Regressão Logística na mesma pipeline, reduzindo o risco de divergência entre treinamento e inferência.

## Vídeo de apresentação

A apresentação do projeto foi estruturada com base na metodologia **STAR** — Situação, Tarefa, Ação e Resultado — e demonstra as principais etapas do desenvolvimento, os resultados dos modelos e o funcionamento da API.

▶️ [Assistir ao vídeo de apresentação no YouTube](https://youtu.be/pLkH5l2uMoo)

> O vídeo está publicado como não listado e pode ser acessado diretamente pelo link acima.

## Estrutura do repositório

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── ml_canvas.png
│   └── MODEL_CARD.md
├── models/
│   ├── logistic_regression_features.txt
│   └── logistic_regression_final.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_mlp_classifier.ipynb
│   └── 05_cross_validation.ipynb
├── reports/
├── results/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── model_service.py
│   │   └── schemas.py
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── main.py
│   ├── model_evaluation.py
│   ├── model_predictor.py
│   ├── model_trainer.py
│   └── preprocessing.py
├── tests/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_health.py
│   │   └── test_predict.py
│   ├── test_data_validation.py
│   ├── test_model_predictor.py
│   └── test_preprocessing.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Ordem de execução dos notebooks

Os notebooks devem ser executados na seguinte ordem:

1. `01_eda.ipynb`: realiza a análise exploratória, converte `Total Charges`
   para formato numérico e gera `data/processed/dataset.parquet`;
2. `02_baseline_logistic_regression.ipynb`: treina e avalia o baseline de
   Regressão Logística;
3. `03_random_forest.ipynb`: treina e avalia o Random Forest;
4. `04_mlp_classifier.ipynb`: treina e avalia o MLPClassifier;
5. `05_cross_validation.ipynb`: aplica validação cruzada, compara os três
   modelos, seleciona o campeão e gera os artefatos finais.

Os notebooks 02 a 05 dependem do arquivo
`data/processed/dataset.parquet`, criado pelo notebook de EDA.
Alternativamente, o fluxo modular pode ser executado diretamente com
`uv run python -m src.main`, pois ele parte da planilha original, prepara os
dados e gera os artefatos necessários.

## Pré-requisitos

- Python 3.12;
- [uv](https://docs.astral.sh/uv/) para dependências e ambiente virtual;
- Docker, apenas para execução em container.

## Instalação local

Clone o repositório e entre no diretório do projeto:

```bash
git clone https://github.com/alinepmarcondes/TechChallenge1.git
cd TechChallenge1
```

Instale as dependências registradas no arquivo de lock:

```bash
uv sync --frozen
```

Caso o lock ainda não tenha sido gerado ou atualizado:

```bash
uv lock
uv sync
```

## Treinamento

Planilha original em:

```text
data/raw/Telco_customer_churn.xlsx
```

Execute o fluxo modular:

```bash
uv run python -m src.main
```

O processo valida e prepara os dados, realiza a divisão estratificada, treina a pipeline de Regressão Logística e salva o modelo e as métricas.

## Execução da API localmente

O caminho padrão esperado pela API é:

```text
models/logistic_regression_final.joblib
```

Para usar outro caminho, defina a variável `CHURN_MODEL_PATH`:

```bash
export CHURN_MODEL_PATH=models/meu_modelo.joblib
```

Inicie a API:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Recursos disponíveis:

- documentação Swagger: <http://localhost:8000/docs>;
- verificação da API e do modelo: `GET /health`;
- previsão de churn: `POST /predict`.

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.50,
    "total_charges": 1074.00,
    "cltv": 3200
  }'
```

Resposta esperada:

```json
{
  "previsao_churn": 1,
  "probabilidade_churn": 0.7729972826297744
}
```

## Execução com Docker

Construa a imagem na raiz do repositório:

```bash
docker build -t churn-api:latest .
```

Execute o container:

```bash
docker run --rm --name churn-api -p 8000:8000 churn-api:latest
```

Teste o serviço:

```bash
curl http://localhost:8000/health
```

Para acompanhar os logs em outro terminal:

```bash
docker logs -f churn-api
```

O `Dockerfile` copia o modelo para a imagem. Portanto, o arquivo `models/logistic_regression_final.joblib` deve existir antes do `docker build`.

## Testes

Execute toda a suíte:

```bash
uv run pytest
```

Para uma saída mais detalhada:

```bash
uv run pytest -v
```

Os testes abrangem:

- validação do dataset e do alvo;
- conversão de `Total Charges` sem alteração do DataFrame original;
- separação do alvo e descarte das colunas de vazamento;
- carregamento e inferência com o modelo persistido;
- contrato, validação e códigos de status dos endpoints `/health` e `/predict`.

## Reprodutibilidade

- `random_state=42` na divisão treino/teste e nos estimadores aplicáveis;
- validação cruzada estratificada com cinco folds;
- pipeline única para pré-processamento e classificação;
- dependências declaradas em `pyproject.toml` e fixadas em `uv.lock`;
- modelo final versionado como artefato Joblib;
- imagem Docker construída a partir do mesmo código e artefato.


## Limitações e uso responsável

- o modelo foi desenvolvido com um único dataset histórico de telecomunicações;
- o recall de teste de `0.5749` indica que parte dos clientes que cancelam não é identificada;
- o limiar de classificação permanece em `0.5` e não foi otimizado com custos reais de retenção;
- não foi realizada busca sistemática de hiperparâmetros;
- o desempenho pode se degradar se o perfil dos clientes ou os processos comerciais mudarem;
- categorias e faixas não observadas no desenvolvimento exigem monitoramento;
- `CLTV` só deve ser usado se estiver disponível e for calculado de forma equivalente no momento da inferência;
- a predição deve apoiar, e não substituir, a análise humana;
- o modelo não deve ser usado para negar serviços, aplicar tratamento discriminatório ou tomar decisões de alto impacto.

Detalhes adicionais estão no [Model Card](docs/MODEL_CARD.md).

## Próximos passos

- calibrar ou otimizar o limiar de decisão com base no custo de falsos positivos e falsos negativos;
- avaliar busca de hiperparâmetros dentro da validação cruzada;
- analisar importância e explicabilidade das variáveis;
- monitorar qualidade dos dados, drift e desempenho após implantação;
- adicionar autenticação, observabilidade e limites de requisição antes de uso em produção;
- automatizar testes e build da imagem com integração contínua;
- avaliar deploy opcional em AWS, Azure ou GCP.

## Entregáveis atendidos

- notebooks de EDA, baseline e comparação de modelos;
- Regressão Logística, Random Forest e MLPClassifier treinados;
- validação cruzada e tabela comparativa;
- modelo campeão salvo em Joblib;
- código produtivo modularizado em `src/`;
- testes automatizados com Pytest;
- API FastAPI com `/health` e `/predict`;
- execução em container Docker;
- README de entrega e Model Card.

## Autoria

Projeto acadêmico desenvolvido por **Aline Patta Marcondes**.