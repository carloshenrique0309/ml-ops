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
uv run dvc push
```

## MLflow

```bash
uv run python -m churn.train
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O treino registra parâmetros, métricas, modelo e tags de linhagem com o commit do
Git e o hash do dado versionado pelo DVC. O backend local usa SQLite em
`mlflow.db`.

## Data Pipeline

O pipeline segue a ideia de data pipeline descrita no ml-ops.org: ingestão,
validação, limpeza/transformação e separação para treino/avaliação antes do
modelo.
