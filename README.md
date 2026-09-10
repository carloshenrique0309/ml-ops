
ML Ops
======

Clone o repositorio:

```bash
git clone https://github.com/carloshenrique0309/ml-ops.git
cd ml-ops
```

Rode o modelo de churn:

```bash
uv run --directory churn python -m churn.train
```

O treino usa `churn/data/churn.csv` e salva o modelo em
`churn/modelo_final_v3_ok.pkl`.
