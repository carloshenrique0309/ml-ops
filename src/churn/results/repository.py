"""Leitura e consulta dos resultados (PostgreSQL ou arquivo CSV/Parquet).

Nenhum modelo é carregado: a fonte da verdade é a tabela do banco (ou um
arquivo gerado pelo job de batch scoring).
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_ALIASES = ("customer_id", "customerid", "client_id", "clientid", "id")
PROBABILITY_ALIASES = ("churn_probability", "churn_proba", "probability", "proba", "score")
PREDICTION_ALIASES = ("churn_prediction", "predicted_churn", "prediction", "churn")

TRUTHY = {"1", "true", "yes", "y", "sim", "churn", "churned", "em risco"}
FALSY = {"0", "false", "no", "n", "nao", "não", "no churn", "sem risco"}

ResultsSource = Literal["postgres", "csv"]


class ResultsError(RuntimeError):
    """Erro base do viewer."""


class ResultsUnavailableError(ResultsError):
    """Fonte ausente, vazia ou ilegível."""


class ResultsSchemaError(ResultsError):
    """Fonte sem as colunas necessárias."""


class ResultsRepository:
    """Mantém os resultados em memória e responde consultas simples.

    A fonte padrão é uma tabela PostgreSQL. Para rodar localmente sem banco,
    use ``source="csv"`` e informe ``path``.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        source: ResultsSource = "postgres",
        database_url: str | None = None,
        schema: str = "public",
        table: str = "churn_results",
        sslmode: str | None = None,
        id_column: str | None = None,
        probability_column: str | None = None,
        prediction_column: str | None = None,
        churn_threshold: float = 0.5,
    ) -> None:
        self.source = source
        self.path = Path(path) if path is not None else None
        self.database_url = database_url
        self.schema = schema
        self.table = table
        self.sslmode = sslmode
        self.churn_threshold = churn_threshold

        self._lock = threading.RLock()
        self._frame: pd.DataFrame | None = None
        self._updated_at: datetime | None = None
        self._id_override = id_column
        self._probability_override = probability_column
        self._prediction_override = prediction_column
        self._id_column: str | None = None
        self._probability_column: str | None = None
        self._prediction_column: str | None = None

    # ---- Estado ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._frame is not None

    @property
    def total_rows(self) -> int:
        return 0 if self._frame is None else int(len(self._frame))

    @property
    def columns(self) -> list[str]:
        return [] if self._frame is None else [str(column) for column in self._frame.columns]

    @property
    def loaded_at(self) -> datetime | None:
        return self._updated_at

    @property
    def id_column(self) -> str | None:
        return self._id_column

    @property
    def source_label(self) -> str:
        """Descrição da fonte sem expor credenciais."""
        if self.source == "postgres":
            prefix = f"{self.schema}." if self.schema else ""
            return f"postgres://{prefix}{self.table}"
        return "" if self.path is None else str(self.path)

    # ---- Carga ----------------------------------------------------------
    def load(self) -> None:
        """(Re)lê a fonte e revalida as colunas."""
        with self._lock:
            frame = self._read()
            columns = [str(column) for column in frame.columns]

            id_column = self._pick(frame, ID_ALIASES, self._id_override)
            if id_column is None:
                raise ResultsSchemaError(
                    "Nenhuma coluna de identificação encontrada. Esperado uma destas: "
                    f"{', '.join(ID_ALIASES)}. Colunas disponíveis: {', '.join(columns)}"
                )
            probability_column = self._pick(
                frame, PROBABILITY_ALIASES, self._probability_override
            )
            prediction_column = self._pick(
                frame, PREDICTION_ALIASES, self._prediction_override
            )

            # Só substitui o estado depois de validar tudo.
            self._frame = frame
            self._id_column = id_column
            self._probability_column = probability_column
            self._prediction_column = prediction_column
            self._updated_at = self._resolve_updated_at()

            logger.info(
                "Resultados carregados de %s: %s linhas (id=%s, prob=%s, pred=%s)",
                self.source_label,
                len(frame),
                self._id_column,
                self._probability_column,
                self._prediction_column,
            )

    def _resolve_updated_at(self) -> datetime:
        if self.source == "csv" and self.path is not None and self.path.exists():
            return datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _read(self) -> pd.DataFrame:
        if self.source == "postgres":
            return _read_postgres_frame(
                self.database_url, self.schema, self.table, self.sslmode
            )
        return self._read_file()

    def _read_file(self) -> pd.DataFrame:
        if self.path is None:
            raise ResultsUnavailableError(
                "RESULTS_PATH não configurado. Informe o arquivo ou use RESULTS_SOURCE=postgres."
            )
        if not self.path.exists():
            raise ResultsUnavailableError(
                f"Arquivo de resultados não encontrado: {self.path}. "
                "Ajuste RESULTS_PATH ou gere o arquivo do batch scoring."
            )

        suffix = self.path.suffix.lower()
        try:
            if suffix in {".parquet", ".pq"}:
                frame = pd.read_parquet(self.path)
            elif suffix in {".csv", ".txt"}:
                frame = pd.read_csv(self.path)
            else:
                raise ResultsUnavailableError(
                    f"Formato não suportado: {suffix or '(sem extensão)'}. Use .csv ou .parquet."
                )
        except ResultsUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ResultsUnavailableError(f"Falha ao ler {self.path}: {exc}") from exc

        if frame.empty:
            raise ResultsUnavailableError(f"O arquivo de resultados está vazio: {self.path}")
        return frame

    def _pick(
        self, frame: pd.DataFrame, aliases: tuple[str, ...], override: str | None
    ) -> str | None:
        if override:
            if override not in frame.columns:
                raise ResultsSchemaError(
                    f"Coluna {override!r} não existe na fonte. "
                    f"Disponíveis: {', '.join(str(column) for column in frame.columns)}"
                )
            return override

        by_lower = {str(column).strip().lower(): str(column) for column in frame.columns}
        for alias in aliases:
            if alias in by_lower:
                return by_lower[alias]
        return None

    def _require_frame(self) -> pd.DataFrame:
        if self._frame is None:
            raise ResultsUnavailableError(
                "Resultados não carregados. Verifique DATABASE_URL/RESULTS_PATH "
                "e chame POST /api/reload."
            )
        return self._frame

    # ---- Consultas ------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        frame = self._require_frame()
        total = int(len(frame))
        mask = self._churn_mask()
        churn = int(mask.fillna(False).sum()) if mask is not None else None

        average: float | None = None
        if self._probability_column:
            value = self._probability_series().mean()
            average = None if pd.isna(value) else float(value)

        return {
            "total": total,
            "churn": churn,
            "not_churn": None if churn is None else total - churn,
            "churn_rate": None if churn is None or total == 0 else churn / total,
            "average_probability": average,
            "churn_threshold": self.churn_threshold,
            "id_column": self._id_column,
            "probability_column": self._probability_column,
            "prediction_column": self._prediction_column,
            "columns": self.columns,
            "source": self.source_label,
            "updated_at": self._updated_at,
        }

    def query(
        self,
        *,
        q: str | None = None,
        prediction: Literal["churn", "no_churn"] | None = None,
        min_probability: float | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        frame = self._require_frame()
        mask = pd.Series(True, index=frame.index)

        if q:
            text = frame.astype("string")
            matches = text.apply(
                lambda column: column.str.contains(q, case=False, na=False, regex=False)
            )
            mask &= matches.any(axis=1)

        if prediction is not None:
            churn_mask = self._churn_mask()
            if churn_mask is None:
                raise ResultsSchemaError(
                    "Filtro por predição exige uma coluna de predição ou de probabilidade."
                )
            mask &= churn_mask.fillna(False) if prediction == "churn" else ~churn_mask.fillna(False)

        if min_probability is not None and self._probability_column:
            mask &= self._probability_series() >= min_probability

        filtered = frame.loc[mask]
        if self._probability_column:
            filtered = filtered.sort_values(
                self._probability_column, ascending=False, na_position="last"
            )

        total = int(len(filtered))
        page = filtered.iloc[offset : offset + limit]
        items = [_json_safe(record) for record in page.to_dict(orient="records")]
        return total, items

    def get(self, customer_id: str) -> dict[str, Any] | None:
        frame = self._require_frame()
        values = frame[self._id_column].astype("string")
        matches = frame.loc[values == str(customer_id)]
        if matches.empty:
            return None
        return _json_safe(matches.iloc[0].to_dict())

    def search_customers(self, term: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Busca clientes por trecho do código, no formato usado pela tela inicial."""
        term = (term or "").strip()
        if not term:
            return []

        frame = self._require_frame()
        if self._id_column is None:
            raise ResultsSchemaError(
                "Nenhuma coluna de identificação disponível para a busca."
            )

        values = frame[self._id_column].astype("string")
        mask = values.str.contains(term, case=False, na=False, regex=False)
        filtered = frame.loc[mask]
        if self._probability_column:
            filtered = filtered.sort_values(
                self._probability_column, ascending=False, na_position="last"
            )

        page = filtered.head(limit)
        return [_customer_payload(record) for record in page.to_dict(orient="records")]

    # ---- Auxiliares -----------------------------------------------------
    def _probability_series(self) -> pd.Series:
        frame = self._require_frame()
        return pd.to_numeric(frame[self._probability_column], errors="coerce")

    def _churn_mask(self) -> pd.Series | None:
        if self._prediction_column:
            frame = self._require_frame()
            return frame[self._prediction_column].map(_to_bool).astype("boolean")
        if self._probability_column:
            return self._probability_series() >= self.churn_threshold
        return None


def _read_postgres_frame(
    database_url: str | None,
    schema: str,
    table: str,
    sslmode: str | None = None,
) -> pd.DataFrame:
    """Lê a tabela inteira do PostgreSQL para um DataFrame."""
    if not database_url:
        raise ResultsUnavailableError(
            "DATABASE_URL não configurada. Defina a variável apontando para o "
            "PostgreSQL (ou use RESULTS_SOURCE=csv para ler um arquivo)."
        )
    if not table.strip():
        raise ResultsSchemaError("DB_TABLE não pode ser vazio.")

    try:
        import psycopg2
        from psycopg2 import sql
    except ModuleNotFoundError as exc:  # pragma: no cover - ambiente sem driver
        raise ResultsUnavailableError(
            "Driver PostgreSQL ausente. Instale com: pip install psycopg2-binary"
        ) from exc

    dsn = _with_sslmode(database_url, sslmode)
    identifier = (
        sql.Identifier(schema, table)
        if schema.strip()
        else sql.Identifier(table)
    )
    query = sql.SQL("SELECT * FROM {}").format(identifier)

    try:
        with closing(psycopg2.connect(dsn, connect_timeout=10)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [description[0] for description in cursor.description or []]
                rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise ResultsUnavailableError(
            f"Falha ao ler a tabela {schema}.{table} no PostgreSQL: {exc}"
        ) from exc

    if not columns:
        raise ResultsUnavailableError(
            f"A consulta na tabela {schema}.{table} não retornou nenhuma coluna."
        )

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise ResultsUnavailableError(
            f"A tabela {schema}.{table} está vazia. Importe os resultados antes de iniciar."
        )
    return frame


def _with_sslmode(database_url: str, sslmode: str | None) -> str:
    """Normaliza ``postgres://`` e adiciona ``sslmode`` sem sobrescrever a URL."""
    dsn = database_url
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://") :]
    if not sslmode:
        return dsn

    parts = urlsplit(dsn)
    if not parts.scheme:
        return dsn
    query = dict(parse_qsl(parts.query))
    query.setdefault("sslmode", sslmode)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _customer_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Converte uma linha da fonte para o contrato usado pela tela de busca."""
    lowered = {str(key).strip().lower(): value for key, value in record.items()}

    customer_id = _first_value(lowered, ("customerid", "customer_id", "client_id", "clientid", "id"))
    tenure = _as_number(_first_value(lowered, ("tenure", "tempo_como_cliente")))
    contract = _first_value(lowered, ("contract", "contrato"))
    monthly = _as_number(
        _first_value(lowered, ("monthlycharges", "monthly_charges", "cobranca_mensal"))
    )
    churn_raw = _first_value(
        lowered, ("churn", "churn_prediction", "predicted_churn", "prediction")
    )
    probability = _as_number(
        _first_value(lowered, ("probability", "churn_probability", "churn_proba", "proba", "score"))
    )
    if probability is not None and probability <= 1.0:
        probability *= 100.0

    churn = _to_bool(churn_raw)
    return {
        "customerID": None if customer_id is None else str(customer_id),
        "tenure": 0 if tenure is None else int(tenure),
        "contract": "" if contract is None else str(contract),
        "monthlyCharges": 0.0 if monthly is None else float(monthly),
        "churn": bool(churn) if churn is not None else None,
        "probability": 0.0 if probability is None else float(probability),
    }


def _first_value(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _as_number(value: Any) -> float | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _to_bool(value: Any) -> bool | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if 0.0 <= number <= 1.0:
            return number >= 0.5
        return number != 0

    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return None


def _json_safe(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _clean(value) for key, value in record.items()}


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    # Tipos específicos do PostgreSQL (UUID, JSON, interval...) viram texto.
    return str(value)
