# Churn

Pipeline tabular de churn com validação Pandera, treino scikit-learn e tracking no MLflow.

## Executar

```powershell
uv sync
uv run churn-train
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

O treino valida `data/churn.csv` antes da limpeza, registra parâmetros, MD5 do CSV,
commit Git, accuracy e ROC AUC, e salva o modelo com signature e exemplo de entrada.
O banco de tracking é local (`mlruns.db`).

## Promover o melhor modelo

```powershell
uv run churn-promote
```

O comando escolhe o maior ROC AUC no experimento `churn`, bloqueia resultados abaixo
de 0,80, registra o modelo no registry `churn` e atualiza o alias `@champion`.
O consumidor pode carregar `models:/churn@champion`.

## DVC e Google Drive

O remote do Google Drive ainda precisa ser configurado. É necessário o ID da pasta de
destino no Drive e autenticação Google válida; não coloque credenciais no repositório.
