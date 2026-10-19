# Churn

Projeto de MLOps para classificação de churn.

## Docker

```bash
docker compose up --build
```

## Validação de dados

O arquivo `src/churn/schema.py` define o `ChurnSchema` com tipos, faixas,
valores nulos permitidos e categorias válidas. A carga dos dados usa Pandera com
`lazy=True`, então múltiplos problemas aparecem juntos na mesma falha.

## DVC

```bash
uv run dvc repro
```

Nenhum remote DVC está configurado neste branch. Configure o remote e a autenticação
na máquina da faculdade antes de executar `dvc pull` ou `dvc push`.

## MLflow

```bash
uv run python -m churn.train
uv run python -m churn.experiments
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O treino registra parâmetros, métricas, modelo e tags de linhagem com o commit do
Git e o hash do dado versionado pelo DVC. O backend local usa SQLite em
`mlflow.db`.

## Registro do modelo

```bash
uv run churn-promote
```

O comando escolhe o run com melhor ROC AUC, bloqueia a promoção abaixo de 0,80 e
atualiza o alias `models:/churn@champion` no registry. Cada run de treino registra
signature, input example e os flavors scikit-learn e pyfunc.

## Data Pipeline

O pipeline segue a ideia de data pipeline descrita no ml-ops.org: ingestão,
validação, limpeza/transformação e separação para treino/avaliação antes do
modelo.
