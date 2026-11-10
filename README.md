# churn — MLOps

Projeto de predição de churn com pipeline de MLOps (DVC + MLflow), uma API FastAPI
para servir o **melhor modelo** registrado e um viewer dos resultados já calculados
(sem inferência, front + back na mesma imagem).

## Estrutura

```
src/churn/
├── config.py                  # Settings (pydantic-settings / .env)
├── services/
│   └── model_service.py       # resolve, carrega e executa o modelo
└── api/
    ├── main.py                # create_app(), lifespan e entrypoint
    ├── deps.py                # dependências das rotas
    ├── schemas.py             # contratos Pydantic
    └── routes/
        ├── health.py          # GET /health
        ├── model.py           # GET /model/info, POST /model/reload
        └── predict.py         # POST /predict, POST /predict/batch

src/churn/results/             # viewer de resultados (sem modelo)
├── main.py                    # API + frontend na mesma aplicação
├── repository.py              # leitura e consulta (PostgreSQL ou CSV/Parquet)
├── load_postgres.py           # importação CSV/Parquet -> PostgreSQL
├── config.py
├── schemas.py
└── static/index.html          # frontend (HTML/CSS/JS, sem build)

tests/test_api.py              # testes da API de inferência
tests/test_results_api.py      # testes do viewer de resultados
```

## Requisitos

- Python **3.12+**
- Um MLflow Tracking Server acessível (local ou remoto)

## Setup local

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

Copy-Item .env.example .env
# ajuste MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME etc.
```

> `requirements.txt` é um freeze de outro ambiente (inclusive com um `-e file:///...`
> apontando para outra máquina). Para instalar, use o `pyproject.toml` como acima.

## Subir a API

```powershell
churn-api
# ou
uvicorn churn.api.main:app --reload
```

- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

## Teste rápido sem o pipeline real

Se ainda não existe modelo registrado, este script treina um modelo sintético,
registra no MLflow local (`sqlite:///mlflow.db`) e aponta o alias `champion`:

```powershell
python scripts/demo_local_model.py

$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
churn-api
```

Depois é só chamar `/model/info` e `/predict` (o modelo de exemplo espera
`tenure` e `MonthlyCharges`).

## Como o "melhor modelo" é escolhido

O `ModelService` resolve o modelo nesta ordem:

1. **`MLFLOW_MODEL_URI`** — override explícito (ex.: `models:/churn@champion`);
2. **Alias** do Model Registry (`MLFLOW_MODEL_ALIAS`, padrão `champion`);
3. **Estágio** do Model Registry (`MLFLOW_MODEL_STAGE`, ex.: `Production`);
4. **Versão mais recente** registrada com o nome `MLFLOW_MODEL_NAME`;
5. **Melhor run do experimento** (`MLFLOW_EXPERIMENT_NAME`) ordenado por
   `MODEL_SELECTION_METRIC` (`max` ou `min`), testando até `MODEL_SELECTION_TOP_K`
   runs até encontrar um com o artefato `model`.

Se nada for encontrado, a API sobe em estado **degraded** (`/health` retorna
`status: "degraded"`) e as rotas de predição respondem `503`.

## Endpoints

| Método | Rota             | Descrição                                                        |
| ------ | ---------------- | ---------------------------------------------------------------- |
| GET    | `/health`        | Estado da API e se o modelo está carregado                       |
| GET    | `/model/info`    | Apenas métricas e hiperparâmetros do modelo em serving           |
| POST   | `/model/reload`  | Recarrega o melhor modelo sem reiniciar a API                    |
| POST   | `/predict`       | Predição de um registro                                          |
| POST   | `/predict/batch` | Predição de até 1000 registros                                   |

### Exemplos

```bash
curl http://localhost:8000/model/info
```

```json
{
  "metrics": {"roc_auc": 0.9264},
  "params": {"modelo": "logistic_regression"}
}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure": 12, "MonthlyCharges": 70.35, "Contract": "Month-to-month"}}'
```

```json
{
  "predictions": [1],
  "probabilities": [0.83],
  "model": {
    "name": "churn",
    "version": "3",
    "alias": "champion",
    "uri": "models:/churn@champion"
  }
}
```

```bash
curl -X POST http://localhost:8000/model/reload -H "X-Reload-Token: troque-me"
```

Os campos de `features` são livres de propósito: o modelo recebe um `DataFrame`
com os nomes usados no treino (a ordem é ajustada pela assinatura MLflow, quando
existir). Se o pipeline espera colunas específicas, elas precisam estar no JSON.

`probabilities` só é preenchido quando o modelo expõe `predict_proba` (modelos
`sklearn` são carregados pelo flavor nativo justamente por isso) e a classificação
é binária — nesse caso é a probabilidade da classe positiva.

## Configuração (`.env`)

| Variável                  | Padrão                  | Descrição                                       |
| ------------------------- | ----------------------- | ----------------------------------------------- |
| `MLFLOW_TRACKING_URI`     | `http://localhost:5000` | URI do tracking server                          |
| `MLFLOW_EXPERIMENT_NAME`  | `churn`                 | Experimento usado no fallback                   |
| `MLFLOW_MODEL_NAME`       | `churn`                 | Nome do modelo no registry                      |
| `MLFLOW_MODEL_ALIAS`      | `champion`              | Alias preferido                                 |
| `MLFLOW_MODEL_STAGE`      | —                       | Estágio (usado se o alias não existir)          |
| `MLFLOW_MODEL_URI`        | —                       | URI explícito (tem prioridade sobre os demais)  |
| `MODEL_SELECTION_METRIC`  | `roc_auc`               | Métrica do fallback por melhor run              |
| `MODEL_SELECTION_MODE`    | `max`                   | `max` ou `min`                                  |
| `MODEL_SELECTION_TOP_K`   | `5`                     | Runs candidatos avaliados no fallback           |
| `AUTOLOAD_MODEL`          | `true`                  | Carregar o modelo no startup                    |
| `FAIL_FAST_ON_STARTUP`    | `false`                 | Falhar o startup se o modelo não carregar       |
| `RELOAD_TOKEN`            | —                       | Exige `X-Reload-Token` em `/model/reload`       |
| `LOG_LEVEL`               | `INFO`                  | Nível de log                                    |
| `HOST` / `PORT`           | `0.0.0.0` / `8000`      | Bind do `churn-api`                             |

## Testes

```powershell
pytest
```

Os testes injetam um serviço falso — não precisam de MLflow nem de rede.

## Docker

```powershell
docker build -t churn-api .
docker run --rm -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -e MLFLOW_MODEL_NAME=churn `
  churn-api
```

## Viewer de resultados (sem inferência)

`churn-results` serve **somente consulta** aos resultados já calculados pelo batch
scoring. Não importa MLflow e não carrega modelo: lê uma tabela **PostgreSQL**
(padrão) ou um CSV/Parquet (modo local) e entrega a mesma aplicação como API JSON
e página web (uma única imagem Docker, front + back).

```powershell
# Rodando local com CSV (sem banco)
$env:RESULTS_SOURCE = "csv"
$env:RESULTS_PATH = "data/sample_predictions.csv"
churn-results

# Rodando contra o PostgreSQL
$env:RESULTS_SOURCE = "postgres"
$env:DATABASE_URL = "postgresql://usuario:senha@host:5432/banco"
$env:DB_TABLE = "churn_results"
churn-load-results data/sample_predictions.csv   # importa o CSV para a tabela
churn-results
```

- Página: <http://localhost:8000>
- Docs: <http://localhost:8000/docs>

### Variáveis de ambiente

| Variável        | Padrão                          | Descrição                                                        |
| --------------- | ------------------------------- | ---------------------------------------------------------------- |
| `RESULTS_SOURCE`| `postgres`                      | `postgres` (produção) ou `csv` (local/testes)                    |
| `DATABASE_URL`  | —                               | URL do PostgreSQL (`postgresql://...`)                           |
| `DB_SCHEMA`     | `public`                        | Schema da tabela                                                 |
| `DB_TABLE`      | `churn_results`                 | Tabela com os resultados                                         |
| `DB_SSLMODE`    | —                               | Opcional (`require`, `prefer`, `disable`...)                     |
| `RESULTS_PATH`  | `data/sample_predictions.csv`   | Arquivo usado quando `RESULTS_SOURCE=csv`                        |
| `CHURN_THRESHOLD` | `0.5`                         | Limiar quando só há probabilidade                                |
| `PAGE_SIZE`     | `25`                            | Tamanho padrão da página                                         |
| `ID_COLUMN`     | detectado                       | Sobrescreve a coluna de identificação                            |
| `PROBABILITY_COLUMN` | detectado                  | Sobrescreve a coluna de probabilidade                            |
| `PREDICTION_COLUMN`  | detectado                  | Sobrescreve a coluna de predição                                 |

### Colunas esperadas

A tabela/arquivo precisa de uma coluna de identificação; probabilidade e predição
são opcionais e detectadas automaticamente:

| Papel         | Nomes aceitos                                                        |
| ------------- | -------------------------------------------------------------------- |
| id            | `customer_id`, `customerid`, `client_id`, `clientid`, `id`           |
| probabilidade | `churn_probability`, `churn_proba`, `probability`, `proba`, `score`  |
| predição      | `churn_prediction`, `predicted_churn`, `prediction`, `churn`         |

Use `ID_COLUMN`, `PROBABILITY_COLUMN` e `PREDICTION_COLUMN` para sobrescrever.

### Endpoints

| Método | Rota                        | Descrição                                                          |
| ------ | --------------------------- | ------------------------------------------------------------------ |
| GET    | `/`                         | Página com resumo, busca e tabela                                  |
| GET    | `/health`                   | Fonte carregada e nº de linhas                                     |
| GET    | `/api/summary`              | Totais, taxa de churn e probabilidade média                        |
| GET    | `/api/results`              | Consulta paginada (`q`, `prediction`, `min_probability`, `limit`, `offset`) |
| GET    | `/api/results/{customer_id}`| Um cliente                                                         |
| POST   | `/api/reload`               | Relê a tabela/arquivo após o batch scoring                         |

### Importar CSV para o PostgreSQL

```powershell
# Com DATABASE_URL definida no ambiente:
churn-load-results data/sample_predictions.csv

# Ou passando a URL explicitamente:
churn-load-results data/sample_predictions.csv `
  --database-url "postgresql://usuario:senha@host:5432/banco" `
  --table churn_results
```

O script **substitui** a tabela por padrão (`--if-exists replace`); use
`--if-exists append` para adicionar linhas.

### Docker (front + back na mesma imagem)

```powershell
docker build -f Dockerfile.results -t churn-results .

# PostgreSQL
docker run --rm -p 8000:8000 `
  -e RESULTS_SOURCE=postgres `
  -e DATABASE_URL="postgresql://usuario:senha@host:5432/banco" `
  -e DB_TABLE=churn_results `
  churn-results

# CSV local (fallback)
docker run --rm -p 8000:8000 `
  -e RESULTS_SOURCE=csv `
  -e RESULTS_PATH=/dados/predictions.parquet `
  -v "${PWD}\data:/dados:ro" `
  churn-results
```

### Deploy no Railway

1. No projeto do Railway, clique em **Add** → **Database** → **PostgreSQL**.
2. No serviço da aplicação, abra **Variables** e adicione:
   - `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (referência ao serviço criado)
   - `RESULTS_SOURCE` = `postgres`
   - `DB_SCHEMA` = `public`
   - `DB_TABLE` = `churn_results`
3. Faça o deploy. O `/health` fica `degraded` até a tabela existir.
4. Importe os resultados usando a **URL pública** do PostgreSQL (aba
   **Connect** do serviço de banco), do seu computador:

   ```powershell
   churn-load-results data/sample_predictions.csv `
     --database-url "postgresql://postgres:senha@containers-....railway.app:..../railway"
   ```

5. Chame `POST /api/reload` (ou reinicie o serviço) para carregar a tabela.

## Próximos passos

- Registrar o modelo no MLflow com alias `champion` ao final do treino.
- Adicionar validação de schema (pandera) dos dados de entrada na API.
- Publicar a imagem no registry e automatizar deploy no CI.
