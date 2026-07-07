"""配置加载。所有路径均通过 configs/config.yaml 注入。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Paths:
    """项目路径集合。所有路径都基于 project_root 解析为绝对路径。"""

    project_root: Path
    data_root: Path
    raw: Path
    normalized: Path
    factor: Path
    label: Path
    evaluation: Path
    daily_bar: Path
    pool_db: Path

    def ensure(self) -> None:
        """确保数据目录存在。"""
        for p in [self.raw, self.normalized, self.factor, self.label, self.evaluation, self.pool_db.parent]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """全局配置。"""

    raw: dict[str, Any] = field(default_factory=dict)
    paths: Paths | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    data_source: dict[str, Any] = field(default_factory=dict)
    factor: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _resolve(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


@lru_cache(maxsize=1)
def load_config() -> Config:
    """加载并合并所有配置文件。结果会被缓存。"""
    main = _load_yaml(CONFIG_DIR / "config.yaml")
    eval_cfg = _load_yaml(CONFIG_DIR / "evaluation.yaml")
    ds_cfg = _load_yaml(CONFIG_DIR / "data_source.yaml")
    fac_cfg = _load_yaml(CONFIG_DIR / "factor.yaml")

    paths_dict = main.get("paths", {})
    root = _resolve(PROJECT_ROOT, paths_dict.get("root", "."))
    paths = Paths(
        project_root=root,
        data_root=_resolve(root, paths_dict.get("data_root", "data")),
        raw=_resolve(root, paths_dict.get("raw", "data/raw")),
        normalized=_resolve(root, paths_dict.get("normalized", "data/normalized")),
        factor=_resolve(root, paths_dict.get("factor", "data/factor")),
        label=_resolve(root, paths_dict.get("label", "data/label")),
        evaluation=_resolve(root, paths_dict.get("evaluation", "data/evaluation")),
        daily_bar=_resolve(root, paths_dict.get("daily_bar", "data/normalized/daily_bar.parquet")),
        pool_db=_resolve(root, paths_dict.get("pool_db", "data/pool/factors.db")),
    )
    return Config(raw=main, paths=paths, evaluation=eval_cfg, data_source=ds_cfg, factor=fac_cfg)
