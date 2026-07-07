"""因子池 CLI：显式登记评价结果、查看、状态流转、持久 leaderboard。

示例：
    python scripts/pool.py register --factor reverse_5d \
        --from data/evaluation/reverse_5d_h5/summary.json --source 自研 --category 反转
    python scripts/pool.py list
    python scripts/pool.py show reverse_5d
    python scripts/pool.py set-status reverse_5d validated --reason "OOS 稳健"
    python scripts/pool.py leaderboard
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from qflab.data.storage import load_factor
from qflab.pool import FactorPool, register as pool_register
from qflab.pool.store import FactorPool as _FP  # noqa: F401 (type hint clarity)

app = typer.Typer(add_completion=False, help="因子池管理")
console = Console()


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


@app.command()
def register(
    factor: str = typer.Option(..., "--factor", help="因子名"),
    from_: str = typer.Option(..., "--from", help="评价 summary.json 路径"),
    source: str = typer.Option("", "--source", help="出处：研报/作者/自研"),
    category: str = typer.Option("", "--category", help="类别：动量/反转/波动..."),
    description: str = typer.Option("", "--description"),
    corr_threshold: float = typer.Option(0.8, "--corr-threshold"),
    no_corr: bool = typer.Option(False, "--no-corr", help="跳过相关性计算"),
):
    """把一次评价结果登记入池(显式)。自动算与池中因子的相关性。"""
    summary = json.loads(Path(from_).read_text(encoding="utf-8"))

    factor_values = None
    if not no_corr:
        pool_preview = FactorPool()
        existing = [r.name for r in pool_preview.list_all()]
        pool_preview.close()
        names = set(existing) | {factor}
        factor_values = {}
        for n in names:
            try:
                factor_values[n] = load_factor(n)
            except FileNotFoundError:
                pass
        if factor not in factor_values:
            console.print(f"[yellow]警告：因子值 {factor} 未找到，跳过相关性计算[/yellow]")
            factor_values = None

    with FactorPool() as pool:
        rec = pool_register(
            pool, summary, factor_values=factor_values,
            source=source, category=category, description=description,
            corr_threshold=corr_threshold,
        )
    console.print(f"[green]已登记[/green] {rec.name} v{rec.version} status={rec.status.value}")
    if rec.max_corr_with:
        console.print(f"  最相关：{rec.max_corr_with} ({rec.max_corr:+.3f})")
    if rec.status_reason:
        console.print(f"  备注：{rec.status_reason}")


@app.command(name="list")
def list_factors_cmd(
    status: str = typer.Option(None, "--status"),
    category: str = typer.Option(None, "--category"),
):
    """列出池中因子。"""
    with FactorPool() as pool:
        recs = pool.list_all(status=status, category=category)
    if not recs:
        console.print("[yellow]因子池为空。用 register 登记因子。[/yellow]")
        return
    table = Table(title=f"因子池 ({len(recs)})")
    for col in ["name", "v", "cat", "status", "rankIC", "ICIR", "OOS_IC", "Sharpe", "maxCorr"]:
        table.add_column(col)
    for r in recs:
        mc = f"{r.max_corr_with}:{r.max_corr:+.2f}" if r.max_corr_with else "-"
        table.add_row(
            r.name, str(r.version), r.category or "-", r.status.value,
            _fmt(r.ic_rank_mean), _fmt(r.icir_rank), _fmt(r.oos_ic_rank),
            _fmt(r.ls_sharpe), mc,
        )
    console.print(table)


@app.command()
def show(name: str):
    """查看单个因子完整元数据。"""
    with FactorPool() as pool:
        rec = pool.get(name)
    if rec is None:
        console.print(f"[red]未找到因子 {name}[/red]")
        raise typer.Exit(1)
    console.print_json(rec.model_dump_json(indent=2))


@app.command(name="set-status")
def set_status(
    name: str,
    status: str = typer.Argument(..., help="candidate|validated|retired"),
    reason: str = typer.Option("", "--reason"),
):
    """更新因子状态。"""
    with FactorPool() as pool:
        rec = pool.set_status(name, status, reason)
    if rec is None:
        console.print(f"[red]未找到因子 {name}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{name} → {status}[/green] {reason}")


@app.command()
def leaderboard(sort_by: str = typer.Option("icir_rank", "--sort-by")):
    """从池导出持久排序榜(按指标绝对值降序)。"""
    with FactorPool() as pool:
        df = pool.to_dataframe()
    if df.empty:
        console.print("[yellow]因子池为空。[/yellow]")
        return
    if sort_by in df.columns:
        df = df.reindex(df[sort_by].abs().sort_values(ascending=False).index)
    table = Table(title=f"因子池 Leaderboard (按 |{sort_by}|)")
    for col in ["name", "category", "status", "ic_rank_mean", "icir_rank", "oos_ic_rank", "ls_sharpe"]:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(
            str(r["name"]), str(r.get("category") or "-"), str(r["status"]),
            _fmt(r.get("ic_rank_mean")), _fmt(r.get("icir_rank")),
            _fmt(r.get("oos_ic_rank")), _fmt(r.get("ls_sharpe")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
