"""Build the offline browser snapshot from the repository's saved evidence."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source(label, path):
    normalized = (ROOT / path).read_text(encoding="utf-8").encode("utf-8")
    return {
        "label": label,
        "url": f"{REPOSITORY}/blob/main/{path}",
        "sha256_lf": hashlib.sha256(normalized).hexdigest(),
    }


def fact(label, value, kind=None):
    return {"label": label, "value": value, "format": kind}


def column(key, title, kind=None):
    return {"key": key, "title": title, "format": kind}


def order(key, title, descending=True):
    return {"key": key, "title": title, "id": key, "descending": descending}


def metric(label, value, note):
    return {"label": label, "value": value, "note": note}


def table(title, columns, rows, *formats):
    return {
        "title": title,
        "table": {"columns": columns, "rows": rows, "formats": formats},
    }


def theme(accent, muted, base, glow):
    return {
        "accent": accent,
        "accent-soft": muted,
        "hero-base": base,
        "hero-glow": glow,
    }


REPOSITORY = "https://github.com/dev-belly/alphaforge"
EXPERIMENT = "docs/experiments/real-sp500-session-v2"


def optional_float(value):
    return float(value) if value else None


def model_seed(value):
    if not value.strip():
        return None
    number = Decimal(value)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"Invalid model seed: {value}")
    return int(number)


def build():
    rows = read_csv(f"{EXPERIMENT}/metrics.csv")
    split = read_json(f"{EXPERIMENT}/split_audit.json")
    names = {
        "equal_weight": "等权基准",
        "single_momentum": "单一动量",
        "ridge": "Ridge",
        "lightgbm": "LightGBM",
        "ridge_without_size_feature": "Ridge（移除规模特征）",
    }
    folds = {}
    for score in read_csv(f"{EXPERIMENT}/folds.csv"):
        model = (score["method"], model_seed(score["seed"]))
        fold = int(score["fold"])
        values = folds.setdefault(model, {})
        if fold in values:
            raise ValueError(f"Duplicate fold {fold} for {model}")
        values[fold] = float(score["rank_ic_mean"])
    expected_folds = {row["fold"] for row in split["folds"]}
    records = []
    for row in rows:
        method = row["method"]
        seed_key = model_seed(row["seed"])
        seed = str(seed_key) if seed_key is not None else "基准"
        cost = int(float(row["cost_multiplier"]))
        identifier = f"{method}-{seed if seed_key is not None else 'baseline'}-{cost}"
        picks = folds.get((method, seed_key), {})
        if seed_key is not None and picks.keys() != expected_folds:
            raise ValueError(f"Incomplete fold metrics for {method} seed {seed}")
        details = [
            fact("净年化收益", float(row["cagr"]), "percent"),
            fact("净年化波动", float(row["ann_vol"]), "percent"),
            fact("净夏普", float(row["sharpe"]), "fixed3"),
            fact("成本拖累（年化收益）", float(row["cost_drag_cagr"]), "percent"),
            fact("平均换手率", float(row["avg_turnover"]), "percent"),
            fact("平均 Rank IC", optional_float(row["rank_ic_mean"]), "fixed4"),
        ]
        records.append(
            {
                "id": identifier,
                "title": f"{names[method]} · {seed} · {cost}×",
                "chartLabel": f"{names[method]} / {seed} / {cost}×",
                "subtitle": f"{row['backtest_start']} 至 {row['backtest_end']} · {row['n_rebalances']} 次再平衡",
                "values": {
                    "method": method,
                    "seed": seed,
                    "cost": cost,
                    "sharpe": float(row["sharpe"]),
                    "cagr": float(row["cagr"]),
                    "drawdown": float(row["max_drawdown"]),
                    "drag": float(row["cost_drag_cagr"]),
                },
                "facts": details,
                "charts": {
                    "folds": [
                        {"label": f"窗口 {fold + 1}", "value": value}
                        for fold, value in sorted(picks.items())
                    ]
                },
                "related": "split",
            }
        )
    fold_rows = [
        [row["fold"] + 1, row["train_end"], row["last_train_label_end"], row["test_start"]]
        for row in split["folds"]
    ]
    return {
        "repository": REPOSITORY,
        "slug": "alphaforge",
        "project": "AlphaForge",
        "mark": "AF",
        "theme": theme("#444fab", "#eef0ff", "#1b2247", "#4e52a5"),
        "workspace": "量化模型研究",
        "title": "策略与成本对照",
        "description": "横向比较基准与模型，切换执行成本和种子，展开交易日窗口与标签截止日期。",
        "dataKind": "冻结市场样本 · 走步复拟合",
        "hero": metric("可比实验组合", str(len(records)), "基准、模型、特征消融与两档成本"),
        "metrics": [
            metric("走步窗口", str(len(fold_rows)), "按市场交易日隔离"),
            metric(
                "预测标签跨度",
                f"{split['label_horizon_sessions']} 个交易日",
                "按实际训练标签截止日期核对",
            ),
            metric("成本档位", "1× / 2×", "对比保存的净收益与成本拖累"),
        ],
        "recordHeading": "实验组合",
        "recordLabel": "方法 · 种子 · 成本",
        "detailHeading": "指标与时间切分",
        "searchPlaceholder": "搜索模型或种子",
        "initialRecord": "ridge-11-1",
        "filters": [
            {"key": "method", "title": "方法", "labels": names},
            {"key": "seed", "title": "种子"},
            {"key": "cost", "title": "执行成本", "labels": {"1": "基准成本", "2": "双倍成本"}},
        ],
        "columns": [
            column("sharpe", "净夏普", "fixed3"),
            column("cagr", "净年化", "percent"),
            column("drawdown", "最大回撤", "percent"),
            column("drag", "成本拖累", "percent"),
        ],
        "orders": [order("sharpe", "净夏普最高"), order("drag", "成本拖累最高")],
        "views": [
            {
                "id": "sharpe",
                "label": "净夏普",
                "title": "策略的风险调整表现",
                "metric": "sharpe",
                "format": "fixed3",
                "descending": True,
                "caption": "显示当前筛选中的高净夏普组合及所选组合。用成本筛选固定同一档假设。",
            },
            {
                "id": "drag",
                "label": "成本拖累",
                "title": "毛收益与净收益的年化差",
                "metric": "drag",
                "format": "percent",
                "descending": True,
                "caption": "显示高拖累组合及所选组合，单位为百分点。",
            },
            {
                "id": "folds",
                "label": "窗口 Rank IC",
                "title": "所选模型的逐窗口 Rank IC",
                "mode": "detail",
                "format": "fixed4",
                "caption": "对照六个测试窗口的样本外秩相关。零线标出正负方向。",
                "empty": "选择预测模型以查看逐窗口 Rank IC。",
            },
        ],
        "related": {
            "split": [
                table("共同走步窗口", ["窗口", "训练截止", "标签结束", "测试开始"], fold_rows),
                {"title": "时间顺序", "text": "训练标签的实际结束日早于每个窗口的测试起点。"},
            ]
        },
        "sourceNote": "真实市场价格快照用于探索性比较。历史指数成分与披露可得时点存在口径缺口，成本按仓库实验假设计算。",
        "sources": [
            source("22 组组合指标", f"{EXPERIMENT}/metrics.csv"),
            source("逐窗口 Rank IC", f"{EXPERIMENT}/folds.csv"),
            source("真实标签边界核查", f"{EXPERIMENT}/split_audit.json"),
            source("研究设计与局限", "docs/research-comparison.md"),
        ],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the generated snapshot")
    args = parser.parse_args()
    document = build()
    records = document.pop("records")
    lines = ",\n".join(
        "    " + json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records
    )
    header = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    result = (
        "// Generated by build_data.py from repository evidence.\n"
        "window.workbenchData = " + header[:-2] + ',\n  "records": [\n' + lines + "\n  ]\n};\n"
    )
    path = DIRECTORY / "data.js"
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != result:
            parser.exit(1, "Snapshot differs. Run build_data.py without --check.\n")
        print(f"Verified {len(records)} records")
    else:
        path.write_text(result, encoding="utf-8", newline="\n")
        print(f"Wrote {len(records)} records")


if __name__ == "__main__":
    main()
