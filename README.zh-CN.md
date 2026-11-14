# AlphaForge｜让量化实验可以复核

[English](README.md) · [完整实验结果](https://dev-belly.github.io/alphaforge/experiments/real-sp500-session-v2/) · [中文研究简报](docs/research-comparison.zh-CN.md)

**一个复杂模型，真的比简单策略更好吗？**

AlphaForge 用同一批数据、预测样本、调仓规则和成本假设，对照等权、固定动量、Ridge、LightGBM，以及去掉显式规模特征的 Ridge。它把数据处理、因子、训练切分、组合交易、成本和研究报告串在一起。

## 本次已实际重跑的结果

2026-11-14，按修正后的交易日隔离重新拟合全部模型。原始来源文件及最终监督数据集的哈希与旧实验一致；保留 84 万条训练数据、24 个特征、约 55 万条共同测试预测。全部 22 个随机种子 / 成本组合都能查看。

| 基础假设成本下 | 净年化收益 | 净 Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: |
| 等权基准 | 12.75% | 1.141 | -18.70% |
| 固定 12–1 动量 | 11.38% | 0.871 | -23.40% |
| Ridge | 12.77% | 1.009 | -21.65% |
| LightGBM，三个种子均值 | 17.07% | 1.252 | -22.10% |
| Ridge 去掉显式规模特征 | 12.43% | 0.970 | -21.57% |

**值得解释的结果：** Ridge 的年化收益与等权接近，但风险调整后的表现更弱；模型复杂度没有自动变成更好的组合。LightGBM 的结果也要同时看成本敏感性、不同种子和回撤。

上述是 2015-01 至 2019-10 的历史模拟结果，扣除声明的模型成本，不代表实际成交收益。当前成员名单缺少完整退出历史，财务披露时点使用假设滞后，规模使用美元成交量代理；这些限制继续保留。

## 六个窗口，不只写一句“防止未来信息泄漏”

新增的 `split_audit.json` 保存原市场日历、监督样本日历、每个窗口的样本数，以及训练结束、标签结束、测试开始的日期。比较模型拟合前检查：**训练标签的 21 个交易日收益区间，必须严格早于测试开始时点结束。**

例如第一折：训练观察到 **2013-11-29**，最后一条训练标签在 **2013-12-31** 结束，测试从 **2014-01-02** 开始。周末和休市不占用缓冲期；监督数据缺日也不会偷偷改变标签期限。

[查看窗口记录](docs/experiments/real-sp500-session-v2/split_audit.json) · [检查实现](src/alphaforge/models/audit.py) · [独立重算保存结果](src/alphaforge/comparison.py)

## 从核验开始

从仓库根目录运行，需要 Python 3.12 和 Git；首次安装需要网络。

```bash
python -m pip install -e ".[viz]" -c research/requirements-session-refit.txt
python scripts/verify_comparison.py docs/experiments/real-sp500-session-v2
```

核验无需下载市场数据，也不重新训练。它从逐日收益重算组合指标、检查每个实验组合，并重建六个窗口的日期与样本数。完整重训方法和 Windows 虚拟环境命令见[中文简报](docs/research-comparison.zh-CN.md#复现与核验)。

## 继续读代码

| 关注点 | 入口 |
| --- | --- |
| 前推训练、隔离与预测 | [训练管道](src/alphaforge/models/pipeline.py) · [窗口核查](src/alphaforge/models/audit.py) |
| 因子与 Rank-IC | [因子评价口径](docs/modules/factor_research.md) |
| 组合与交易成本 | [回测与执行](docs/modules/backtesting.md) |
| 输入和来源范围 | [数据质量](docs/modules/data.md) |
| 完整接口、测试与架构 | [英文文档](README.md) |
