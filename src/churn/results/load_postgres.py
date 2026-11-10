"""Importa um arquivo de resultados (CSV/Parquet) para uma tabela PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _normalize_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql://" + database_url[len("postgres://") :]
    return database_url


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato não suportado: {suffix or '(sem extensão)'}. Use .csv ou .parquet.")


def load_frame(
    frame: pd.DataFrame,
    *,
    database_url: str,
    table: str = "churn_results",
    schema: str = "public",
    if_exists: str = "replace",
    chunksize: int = 1000,
) -> None:
    """Grava o DataFrame na tabela, criando-a/substituindo-a conforme ``if_exists``."""
    try:
        from sqlalchemy import create_engine
    except ModuleNotFoundError as exc:  # pragma: no cover - ambiente sem SQLAlchemy
        raise RuntimeError(
            "SQLAlchemy não instalado. Rode: pip install sqlalchemy psycopg2-binary"
        ) from exc

    engine = create_engine(_normalize_url(database_url))
    with engine.begin() as connection:
        frame.to_sql(
            table,
            connection,
            schema=schema or None,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Importa resultados de churn (CSV/Parquet) para o PostgreSQL."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=os.getenv("RESULTS_PATH", "data/sample_predictions.csv"),
        help="Arquivo CSV/Parquet (padrão: RESULTS_PATH ou data/sample_predictions.csv).",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="URL do PostgreSQL (padrão: variável DATABASE_URL).",
    )
    parser.add_argument("--table", default=os.getenv("DB_TABLE", "churn_results"))
    parser.add_argument("--schema", default=os.getenv("DB_SCHEMA", "public"))
    parser.add_argument(
        "--if-exists",
        choices=["replace", "append", "fail"],
        default="replace",
        help="replace (padrão) recria a tabela; append adiciona linhas.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="Linhas por INSERT (padrão: 1000)."
    )
    args = parser.parse_args(argv)

    if not args.database_url:
        parser.error(
            "informe --database-url ou defina DATABASE_URL "
            "(Railway: use a URL pública do serviço PostgreSQL)."
        )

    path = Path(args.path)
    if not path.exists():
        parser.error(f"arquivo não encontrado: {path}")

    try:
        frame = _read_frame(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao ler {path}: {exc}", file=sys.stderr)
        return 1

    if frame.empty:
        print(f"O arquivo {path} está vazio; nada a importar.", file=sys.stderr)
        return 1

    destination = f"{args.schema}.{args.table}" if args.schema else args.table
    try:
        load_frame(
            frame,
            database_url=args.database_url,
            table=args.table,
            schema=args.schema,
            if_exists=args.if_exists,
            chunksize=args.chunksize,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao importar para {destination}: {exc}", file=sys.stderr)
        return 1

    print(f"{len(frame)} linha(s) importada(s) de {path} para {destination}.")
    print("Agora reinicie o app ou chame POST /api/reload para recarregar a tabela.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
