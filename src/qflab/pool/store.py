"""因子池 SQLite 存储层。

只存元数据。主键 (name, version)：同名因子定义变更时递增 version 存新版，
否则原地更新最新指标。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger
from .models import FactorRecord, FactorStatus

logger = get_logger(__name__)

# 表字段顺序（与 FactorRecord 对齐）
_COLUMNS = [
    "name", "version", "created_at", "updated_at",
    "definition", "source", "category", "description",
    "horizon", "ic_rank_mean", "icir_rank", "ic_pos_ratio",
    "ls_sharpe", "ls_annual_return", "ls_max_drawdown", "turnover", "oos_ic_rank",
    "data_start", "data_end", "n_stocks", "preprocess", "cost_bps", "eval_config_json",
    "max_corr_with", "max_corr",
    "status", "status_reason",
    "n_trials", "test_id",
]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS factors (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    definition TEXT,
    source TEXT,
    category TEXT,
    description TEXT,
    horizon INTEGER,
    ic_rank_mean REAL,
    icir_rank REAL,
    ic_pos_ratio REAL,
    ls_sharpe REAL,
    ls_annual_return REAL,
    ls_max_drawdown REAL,
    turnover REAL,
    oos_ic_rank REAL,
    data_start TEXT,
    data_end TEXT,
    n_stocks INTEGER,
    preprocess TEXT,
    cost_bps REAL,
    eval_config_json TEXT,
    max_corr_with TEXT,
    max_corr REAL,
    status TEXT,
    status_reason TEXT,
    n_trials INTEGER,
    test_id TEXT,
    PRIMARY KEY (name, version)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FactorPool:
    """因子池：SQLite 元数据存储。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else load_config().paths.pool_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FactorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ read
    def get(self, name: str, version: int | None = None) -> FactorRecord | None:
        """取因子记录。version 为 None 时取最新版本。"""
        if version is None:
            cur = self._conn.execute(
                "SELECT * FROM factors WHERE name=? ORDER BY version DESC LIMIT 1", (name,)
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM factors WHERE name=? AND version=?", (name, version)
            )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def latest_version(self, name: str) -> int:
        cur = self._conn.execute("SELECT MAX(version) AS v FROM factors WHERE name=?", (name,))
        v = cur.fetchone()["v"]
        return int(v) if v is not None else 0

    def list_all(
        self, status: str | None = None, category: str | None = None, latest_only: bool = True
    ) -> list[FactorRecord]:
        """列出因子。latest_only=True 时每个 name 只返回最新版本。"""
        rows = self._conn.execute("SELECT * FROM factors ORDER BY name, version").fetchall()
        records = [self._row_to_record(r) for r in rows]
        if latest_only:
            by_name: dict[str, FactorRecord] = {}
            for r in records:
                if r.name not in by_name or r.version > by_name[r.name].version:
                    by_name[r.name] = r
            records = list(by_name.values())
        if status:
            records = [r for r in records if r.status.value == status]
        if category:
            records = [r for r in records if r.category == category]
        return sorted(records, key=lambda r: r.name)

    def to_dataframe(self, latest_only: bool = True) -> pd.DataFrame:
        recs = self.list_all(latest_only=latest_only)
        if not recs:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame([r.model_dump(mode="json") for r in recs])

    # ----------------------------------------------------------------- write
    def upsert(self, record: FactorRecord) -> FactorRecord:
        """落库。同名因子：definition 变更 → 新版本；否则更新当前最新版本的指标。"""
        existing = self.get(record.name)  # 最新版本
        now = _now()
        if existing is None:
            record.version = 1
            record.created_at = record.created_at or now
            record.updated_at = now
        elif existing.definition != record.definition and record.definition:
            # 定义变更 → 递增版本，存为新行
            record.version = existing.version + 1
            record.created_at = now
            record.updated_at = now
        else:
            # 原地更新最新版本
            record.version = existing.version
            record.created_at = existing.created_at or now
            record.updated_at = now

        vals = self._record_to_row(record)
        placeholders = ",".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO factors ({','.join(_COLUMNS)}) VALUES ({placeholders})", vals
        )
        self._conn.commit()
        logger.info("Pool upsert: %s v%d (status=%s)", record.name, record.version, record.status.value)
        return record

    def set_status(self, name: str, status: str, reason: str = "") -> FactorRecord | None:
        rec = self.get(name)
        if rec is None:
            return None
        rec.status = FactorStatus(status)
        rec.status_reason = reason
        rec.updated_at = _now()
        self._conn.execute(
            "UPDATE factors SET status=?, status_reason=?, updated_at=? WHERE name=? AND version=?",
            (rec.status.value, reason, rec.updated_at, name, rec.version),
        )
        self._conn.commit()
        return rec

    def delete(self, name: str, version: int | None = None) -> int:
        if version is None:
            cur = self._conn.execute("DELETE FROM factors WHERE name=?", (name,))
        else:
            cur = self._conn.execute("DELETE FROM factors WHERE name=? AND version=?", (name, version))
        self._conn.commit()
        return cur.rowcount

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _record_to_row(record: FactorRecord) -> list:
        d = record.model_dump(mode="json")  # status enum -> str
        return [d[c] for c in _COLUMNS]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FactorRecord:
        return FactorRecord(**{k: row[k] for k in row.keys()})
