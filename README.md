# quant-factor-lab

> 本地运行的 A 股日度横截面因子评价系统。研究用途，**不构成投资建议**。

---

## 这是什么

一个 MVP 量化研究脚手架，专注于 **日频** 横截面因子的评价：

```
原始日线 (parquet)
    └── 因子计算 (long format)
        └── 预处理 (winsorize / zscore / 行业+市值中性化)
            └── 与未来收益对齐 → IC、Rank IC、ICIR
                └── 分位分组 → 多空 NAV、Sharpe、回撤、换手率
                    └── HTML 报告 + Streamlit UI
```

第一版**不依赖外部数据源**，可用内置 sample data 直接跑通。

---

## 目录结构

```
quant-factor-lab/
├── configs/                # 全局/数据源/因子/评价 yaml 配置
├── data/                   # 数据落盘根目录（normalized / factor / evaluation ...）
├── scripts/                # CLI 入口
├── src/qflab/
│   ├── data/               # DataProvider 抽象 + akshare/tushare 骨架 + 存储
│   ├── factors/            # Factor 抽象 + 12 个内置因子 + registry
│   ├── labels/             # forward returns（防未来函数）
│   ├── preprocess/         # winsorize / zscore / 中性化
│   ├── evaluation/         # IC / quantile / portfolio / turnover / report
│   ├── visualization/      # plotly 图
│   ├── api/                # FastAPI 骨架
│   ├── ui/                 # Streamlit 页面
│   └── utils/              # 配置/日志/日期
└── tests/                  # pytest
```

---

## 安装

```bash
cd quant-factor-lab
python -m venv .venv && source .venv/bin/activate    # 可选
pip install -e .
# 如果需要接入真实数据源（第一版可选）：
# pip install -e '.[data]'
# 开发依赖：
# pip install -e '.[dev]'
```

要求 Python ≥ 3.11。

---

## 5 分钟跑通 MVP

```bash
# 1) 生成模拟数据（100 只股票 × 600 个工作日）
python scripts/init_sample_data.py

# 2) 计算所有内置因子
python scripts/compute_factors.py --factor all

# 3) 评价 ret_20d，h=5，5 分组
python scripts/evaluate_factor.py \
  --factor ret_20d --horizon 5 --quantiles 5 \
  --preprocess winsorize_quantile,zscore

# 4) 启动 UI
python scripts/run_streamlit.py
```

报告会落到 `data/evaluation/<factor>_h<horizon>/`，包括：

- `summary.json`：核心指标
- `report.html`：含 IC 曲线、累计 IC、分组收益、多空 NAV、换手率
- `ic_pearson.parquet` / `ic_rank.parquet` / `quantile_returns.parquet` / `nav.parquet` / `turnover.parquet` / `long_short.parquet`

---

## 测试

```bash
python -m pytest -q
```

覆盖：forward return shift、IC、quantile 分组、中性化能否去除市值暴露、缺失值健壮性。

---

## 关键约定

### Forward return（防未来函数）

```
label[T] = close[T+n] / close[T] - 1
```

T 日收盘观察因子值后，T 日收盘建仓、T+n 日收盘平仓。

实现见 `src/qflab/labels/forward_returns.py`。等价于：

```python
label = close.shift(-n) / close - 1.0
```

序列尾部 n 行自然为 NaN，会被丢弃，**不会窥探未来**。

### 数据 schema（long format）

| 必填                                                | 可选                                                  |
|-----------------------------------------------------|-------------------------------------------------------|
| trade_date, instrument, open, high, low, close, volume, amount | adj_factor, industry, market_cap, is_st, is_suspended |

### 因子输出

```
trade_date, instrument, factor_name, factor_value
```

---

## 内置因子

价格反转 / 动量类：`ret_5d`、`ret_20d`、`ret_60d`、`reverse_1d`、`reverse_5d`、`close_to_ma20`、`close_to_ma60`
波动率：`volatility_20d`、`volatility_60d`
量能：`volume_mean_20d`、`amount_mean_20d`
市值：`log_market_cap`

---

## 新增一个因子

1. 在 `src/qflab/factors/` 找一个匹配的文件（或新建一个），继承 `Factor`：

```python
# src/qflab/factors/my_factor.py
from .base import Factor

class HighLowSpread20D(Factor):
    name = "high_low_spread_20d"

    def compute(self, df):
        wide_h = df.pivot(index="trade_date", columns="instrument", values="high")
        wide_l = df.pivot(index="trade_date", columns="instrument", values="low")
        wide = (wide_h - wide_l).rolling(20, min_periods=10).mean()
        return self.to_long(wide)
```

2. 在 `src/qflab/factors/registry.py` 的 `_REGISTRY` 中加上新类，或用 `@register` 装饰器。

3. 重新计算并评价：

```bash
python scripts/compute_factors.py --factor high_low_spread_20d
python scripts/evaluate_factor.py --factor high_low_spread_20d --horizon 5
```

---

## 配置说明

所有路径与默认参数都在 `configs/`：

- `config.yaml`：项目根、数据目录
- `evaluation.yaml`：默认 horizon、quantiles、winsor 参数、年化系数
- `factor.yaml`：因子白名单
- `data_source.yaml`：默认 provider、universe 过滤规则

---

## 接入真实数据源

骨架已留好：

- `src/qflab/data/akshare_provider.py`：`AkshareProvider`
- `src/qflab/data/tushare_provider.py`：`TushareProvider`

下一步：

1. 实现 `get_stock_list / get_daily_bar / get_market_cap / get_industry`，输出列名映射到 `REQUIRED_COLUMNS / OPTIONAL_COLUMNS`。
2. 在 provider 内部统一调用 `normalize_daily_bar` 落库。
3. `python scripts/update_daily_data.py --provider tushare --start 2023-01-01 --end 2024-12-31` 即可。

> Tushare 需要在 `.env` 中设置 `TUSHARE_TOKEN`。Akshare 接口经常变动，请以最新文档为准。

---

## 下一步可以怎么扩展

- **真实数据源**：完成 Akshare/Tushare 实现，落库 daily_bar；用历史交易日历替代工作日近似
- **更精细的标签**：用 `T+1 close → T+n close` 的执行口径减少时点上的乐观偏差
- **更多因子**：财务因子、量价合成、换手 Z 等；引入复权（`adj_factor`）后再算价格类因子
- **更现实的多空组合**：加入交易成本（bps）、ST/停牌/上市天数过滤、可交易股票池
- **横截面回归 IC**：除 Pearson/Rank 外，加 Fama-MacBeth 回归与 t 序列分析
- **因子相关性 / IC 衰减**：批量算 IC by horizon 与因子两两 corr，避免重复因子
- **批量评价**：脚本批跑全因子、汇总成 leaderboard
- **API 化**：基于 `qflab.api` 暴露 `/evaluate` 给前端或调度系统

---

## ⚠ 风控声明

- **本系统仅用于研究和因子评价，不构成投资建议**。
- 注意**未来函数**：建仓/平仓时点必须晚于因子观察时点；当前默认 `T → T+n` 假设 T 收盘可成交，若不能请改用 `T+1 close → T+n close` 口径。
- 注意**幸存者偏差**：sample data 不模拟退市；接入真实数据时务必使用包含已退市股票的全样本。
- 注意**停牌/ST/退市/复权**：`is_st`、`is_suspended` 字段需要在 universe 中过滤；价格类因子建议使用前复权。
- 注意**交易成本**：默认未扣除；高换手率因子在加上单边 5–10 bps 后可能完全消失。
- **因子历史表现不代表未来收益**。
