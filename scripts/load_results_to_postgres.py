"""Atalho para importar resultados para o PostgreSQL.

Uso:
    python scripts/load_results_to_postgres.py data/sample_predictions.csv \
        --database-url "postgresql://usuario:senha@host:5432/banco"

Ou, com a variável DATABASE_URL definida:
    python scripts/load_results_to_postgres.py
"""

from churn.results.load_postgres import main

if __name__ == "__main__":
    raise SystemExit(main())
