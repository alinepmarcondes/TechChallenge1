# Model Card - Modelo de Previsão de Churn

## Informações gerais

| Item | Descrição |
|---|---|
| Nome do modelo | Modelo de Previsão de Churn |
| Versão | 1.0.0 |
| Tipo | Classificação binária |
| Algoritmo | Regressão Logística do Scikit-Learn |
| Saída | Classe prevista e probabilidade de churn |
| Classe positiva | `Churn Value = 1` |
| Artefato | Pipeline serializada em Joblib |
| Responsável | `Aline Patta Marcondes` |
| Data da versão | `2026-10-18` |

## Finalidade

O modelo estima a propensão de cancelamento de clientes de uma operadora de telecomunicações. Sua finalidade é apoiar a priorização de análises e campanhas de retenção, permitindo que equipes de negócio identifiquem clientes que podem demandar atenção preventiva.

O modelo é um componente de apoio à decisão. Ele não determina a causa do churn, não garante que o cliente cancelará o serviço e não deve substituir a avaliação humana.

## Usos pretendidos

- priorizar clientes para campanhas de retenção;
- apoiar análises agregadas de risco de churn;
- comparar estratégias de modelagem em contexto acadêmico;
- demonstrar uma pipeline de Machine Learning servida por API REST.

## Usos não recomendados

- negar, limitar ou precificar serviços de forma automática;
- tomar decisões de alto impacto sem revisão humana;
- aplicar o modelo a empresas, regiões ou períodos muito diferentes sem nova validação;
- interpretar a probabilidade como relação causal;
- usar os resultados para tratamento discriminatório;
- utilizar atributos cuja disponibilidade ou forma de cálculo difira do treinamento.

## Dados

O desenvolvimento utilizou o dataset público **Telco Customer Churn**, da IBM, com mais de 7 mil observações. Os registros incluem características demográficas, serviços contratados, informações de contrato, tempo de permanência, cobranças e histórico de churn.

### Variável alvo

`Churn Value`:

- `0`: cliente permaneceu;
- `1`: cliente apresentou churn.

### Variáveis de entrada

A versão servida pela API utiliza 20 atributos:

- perfil: `Gender`, `Senior Citizen`, `Partner`, `Dependents`;
- relacionamento: `Tenure Months`;
- serviços: `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV` e `Streaming Movies`;
- contrato e cobrança: `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges` e `Total Charges`;
- valor do cliente: `CLTV`.

### Colunas removidas

Foram descartadas informações identificadoras, constantes, geográficas redundantes ou associadas a vazamento do alvo:

`CustomerID`, `Churn Label`, `Churn Score`, `Churn Reason`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude` e `Longitude`.

### Pré-processamento

- conversão de `Total Charges` para numérico;
- imputação numérica pela mediana;
- padronização numérica;
- imputação categórica pela moda;
- codificação one-hot das categorias, com remoção de uma categoria nas
  variáveis binárias (`drop="if_binary"`);
- categorias não observadas ignoradas pelo codificador durante a transformação.

O pré-processamento e o classificador são persistidos na mesma pipeline para manter consistência entre treino e inferência.

## Desenvolvimento e seleção

Foram comparadas três famílias de modelos exigidas no projeto:

- modelo linear: Regressão Logística;
- ensemble de árvores: Random Forest;
- rede neural simples: MLPClassifier.

A avaliação utilizou divisão estratificada entre treino e teste e validação cruzada estratificada com cinco folds. A semente `42` foi fixada nos componentes estocásticos aplicáveis.

A Regressão Logística foi escolhida por apresentar o maior F1-score médio de validação e o melhor recall entre os modelos avaliados, mantendo ROC-AUC competitivo, baixa diferença entre treino e validação, menor custo computacional e maior interpretabilidade.

O Random Forest apresentou métricas perfeitas no treino e queda expressiva na validação, um sinal de overfitting na configuração experimentada. O MLPClassifier obteve ROC-AUC próxima à Regressão Logística, mas F1-score e recall inferiores.

## Métricas

### Validação cruzada - média em cinco folds

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Regressão Logística** | **0.8113** | 0.6656 | **0.5806** | **0.6196** | **0.8588** |
| Random Forest | 0.7993 | 0.6521 | 0.5244 | 0.5811 | 0.8446 |
| MLPClassifier | 0.8037 | **0.6708** | 0.5130 | 0.5810 | 0.8512 |

### Dispersão das métricas de validação

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0.0134 | 0.0273 | 0.0433 | 0.0321 | 0.0125 |
| Random Forest | 0.0135 | 0.0355 | 0.0200 | 0.0245 | 0.0112 |
| MLPClassifier | 0.0094 | 0.0272 | 0.0231 | 0.0196 | 0.0125 |

### Modelo final no conjunto de teste

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8034 |
| Precision | 0.6456 |
| Recall | 0.5749 |
| F1-score | 0.6082 |
| ROC-AUC | 0.8488 |

### Interpretação

- A ROC-AUC de `0.8488` indica boa capacidade de ordenação entre clientes com e sem churn no conjunto de teste.
- O recall de `0.5749` mostra que o modelo identifica aproximadamente 57,5% dos casos positivos usando o limiar padrão, deixando de detectar parte relevante dos clientes que cancelam.
- A precision de `0.6456` indica que aproximadamente 64,6% das classificações positivas estão corretas.
- As métricas não representam, sozinhas, impacto financeiro. A utilidade real depende do custo das campanhas, do custo de não identificar um churn e do benefício de uma retenção bem-sucedida.

## Limiar de decisão

O modelo usa o limiar padrão de `0.5`: probabilidades iguais ou superiores a esse valor são classificadas como churn. Esse limiar não foi otimizado segundo custos reais de negócio.

Em um cenário de retenção, pode ser apropriado reduzir o limiar para aumentar o recall, aceitando mais falsos positivos. A escolha deve ser feita com uma matriz de custos e validada em dados não utilizados no ajuste.

## Limitações

- treinamento e avaliação realizados em um único dataset;
- ausência de validação temporal ou externa;
- recall moderado para a classe de churn;
- ausência de busca sistemática de hiperparâmetros;
- ausência de calibração formal das probabilidades;
- possível mudança de desempenho diante de drift de dados ou conceito;
- categorias e distribuições podem não representar outras operadoras ou regiões;
- o uso de `CLTV` exige confirmar disponibilidade, definição e cálculo equivalentes no momento da inferência;
- correlações aprendidas pelo modelo não demonstram causalidade;
- o conjunto de dados pode não conter todas as variáveis relevantes para o comportamento de cancelamento.

## Riscos de viés e considerações éticas

O dataset contém características como gênero, idade aproximada por indicador de pessoa idosa, estado civil e dependentes. Mesmo quando úteis estatisticamente, essas variáveis podem produzir diferenças de desempenho ou impacto entre grupos.

Antes de qualquer uso real, devem ser avaliados pelo menos:

- distribuição das previsões por grupo;
- precision, recall, F1-score e taxas de erro por subgrupo;
- diferenças nas ofertas ou intervenções geradas a partir da previsão;
- necessidade e legitimidade de cada atributo sensível ou proxy;
- conformidade com a LGPD e políticas internas de privacidade.

Não foram fornecidas métricas segmentadas de equidade nesta versão. Portanto, o modelo não deve ser considerado validado quanto à justiça algorítmica.

## Operação e monitoramento

A aplicação disponibiliza:

- `GET /health`: informa se a API iniciou e se o modelo foi carregado;
- `POST /predict`: valida os atributos, executa a pipeline e retorna classe e probabilidade.

Em produção, recomenda-se monitorar:

- campos ausentes, tipos inválidos e categorias novas;
- distribuição de cada feature e das probabilidades previstas;
- taxa de classificações positivas;
- latência, erros e disponibilidade da API;
- métricas após obtenção do desfecho real;
- desempenho e taxas de erro por subgrupo;
- alterações no processo de cálculo de `CLTV` e demais variáveis.

## Critérios sugeridos para reavaliação

O modelo deve ser reavaliado quando ocorrer ao menos uma destas situações:

- alteração relevante nos produtos, contratos ou políticas comerciais;
- surgimento frequente de categorias desconhecidas;
- mudança substancial na distribuição das entradas ou previsões;
- queda das métricas abaixo dos limites definidos pelo negócio;
- mudança no custo relativo de falsos positivos e falsos negativos;
- disponibilidade de novos dados rotulados representativos.

## Reprodutibilidade e versionamento

- versão do modelo: `1.0.0`;
- algoritmo: `LogisticRegression`;
- semente: `42`;
- ambiente: Python 3.12 e dependências registradas em `pyproject.toml`/`uv.lock`;
- formato: Joblib contendo pipeline completa;
- caminho padrão da API: `models/logistic_regression_final.joblib`.

Artefatos Joblib devem ser carregados apenas de fontes confiáveis, pois a desserialização pode executar código malicioso. Também é recomendável registrar checksum, data de treinamento, versão dos dados e commit Git associado a cada versão futura.

## Aprovação

| Papel | Nome | Data |
|---|---|---|
| Desenvolvimento | `Aline Patta Marcondes` | `2026-10-18` |