# AlphaForge 三遍合规审查报告

**日期**: 2026-09-03
**审查依据**: 项目 Master Prompt（50 条硬约束）+ 第 44 条三轮 Code Review
**方法**: 不依赖历史记忆，直接读源码 + 独立扫描代理交叉验证；对代理指出的每个"bug"均亲自读代码核实。
**结论总调**: 核心量化引擎**基本正确且达标**，但**未真正达到 Master Prompt 的"可公开展示"标准**。存在 1 处诚信硬伤、1 处正确性 bug、若干"完整性"缺口。前会话记忆中的"全部完成"属于**表面完成**（本地 `.venv` 完整安装 + 跑通测试，未做第 44 条三轮 Review，也未逐条对照 50 条）。

---

## 第一遍：逐条符合度（对照 Master Prompt 硬约束）

| # | 约束 | 状态 | 证据 | 备注 |
|---|------|------|------|------|
| 1 | DataProvider + 多 adapter + 质量报告 | 部分 | `base.py` / `local.py` / `sample.py` / `vendors.py` / `quality.py` | **缺 Tushare adapter**；其余齐全 |
| 2 | 因子各组 + preprocessing(config) | 已实现 | `factors/*` `preprocessing.py:74-138` | winsorize/zscore/rank/ind+size neutral 全有 |
| 3 | 因子评价 + tear sheet | 已实现 | `evaluation.py` `report.py` | IC/RankIC/ICIR/正IC比/turnover/分位/多空/衰减齐 |
| 4 | ML walk-forward + purge/embargo + 禁 shuffle | 已实现 | `split.py` `estimators.py` | 无 shuffle=True；ridge/en/rf/lgbm 齐 |
| 5 | cvxpy 组合优化 + 约束 | 已实现 | `optimizer.py:39,71` | 方法名无显式 `alpha-optimization`(mean_variance 等价) |
| 6 | 风险模型 Σ=BFBᵀ+D + 分解 | 已实现 | `factor_model.py` | style 六因子 + 边际/成分贡献齐 |
| 7 | 协方差 sample/shrinkage/LW/EWMA | 已实现 | `covariance.py` | 四种齐 |
| 8 | 回测 signal/exec/rebal 分离 + 成本 | **有 bug** | `engine.py:239,275` | 成本漏记收益序列（见第二遍 R1） |
| 9 | 成本模型 + gross/net 透明 | 部分 | `costs.py` | 成本组件齐；**缺 gross vs net Sharpe 并排对比** |
| 10 | 绩效分析 + tear sheet | 已实现 | `metrics.py` | CAGR/Sharpe/Sortino/MaxDD/Calmar/Beta/Alpha/IR/TE/VaR/CVaR/hit 齐；年化公式正确 |
| 11 | 归因 Brinson + factor + stress | 部分 | `brinson.py` `factor.py` | **缺 Stress Testing / scenario shock** |
| 12 | Market Regime 模块 + dashboard | **缺失** | grep 仅 `sample.py:75` `math_utils.py:90` | 无独立 regime 模块 |
| 13 | AI copilot 真实 tool calling | 部分 | `agents/tools.py` | 真调函数（诚信佳）；**命名不符契约 10 函数**；缺 `analyze_market_regime` |
| 14 | FastAPI 端点契约 | 部分 | `main.py` | 缺 `/portfolio/*` `/factors/{name}` `POST /optimize` `POST /agent/query` `POST /backtests` |
| 15 | 测试覆盖 + pytest/ruff | 部分 | `tests/` | 有 optimizer/metrics/risk/covariance/math(unit)+api/pipeline(integration)+determinism(regression)；**缺 data/factor/cost/attribution 专门单测** |
| 16 | Docker + Makefile + docs(mermaid+分模块) + README | 部分 | `Dockerfile` `Makefile` 有；`docs/` 仅 3 页 | **缺 docker-compose.yml、mermaid、分模块文档** |
| 17 | 安全无密钥/.env/.gitignore | 已实现 | grep 无 API_KEY/SECRET；`config.py` 走 env | 合规 |
| 18 | 研究诚信：不伪造 + sample 标注 + 限制说明 | **有硬伤** | `README.md:20` 谎称 tushare；`sample.py:60` SYNTHETIC 标注好 | **README 谎称 tushare = 诚信硬伤**；缺 Limitations/Case Study 章节 |

**整体达标度 ≈ 68%。** 8 项完整 / 8 项部分 / 3 项缺失（regime / stress / docker-compose + 端点契约 + README 章节为"部分/缺失"集中区）。

---

## 第二遍：三轮 Code Review（第 44 条，全部亲自核实）

### R1 — Correctness（正确性）
- **[真实 bug] 回测成本漏记收益序列** `engine.py`
  - step 2（line 233-240）用 pre-trade `nav_t` 算 `rets[date] = nav_t/prev_nav - 1`。
  - step 3（line 261-275）执行交易后 `nav_t = nav_after`（已扣成本），但**从未回写 `rets[date]`**。
  - 结果：equity 曲线跌了成本，returns 序列却不含该日损失 → `cumprod(rets)` ≠ equity 轨迹，net 的 CAGR/Sharpe 由"不含成本日"的 returns 计算 → **净收益指标虚高**。这是正确性 + 诚信双重问题，必须修。
  - 修复：step 3 扣成本后补 `rets[date] = nav_after / prev_nav - 1`（prev_nav 此时仍为前一日值）。
- **[违约非泄漏] `split.py:137/155`** purge/embargo 乘 `1.5`：配置写 21 天，实际禁用 31.5 天。更保守、不造成泄漏，但违反"配置即契约"，应改为配置原值或显式 safety factor 并在 docstring 说明。
- **代理误报 1（已证伪）** `metrics.py:151` `bench_cagr = end ** (ppy/len(b)) - 1` = `end ** (1/years)`，年化**正确**。
- **代理误报 2（已证伪）** `broker.py:200` `realised_weights *= nav/nav_after` 是把权重正确重定基到 post-cost NAV（持仓市值不变），engine:265 用 `weights*nav_after/px` 还原股数，两者一致，**非双重缩放**。

### R2 — Quant Research Integrity（量化诚信）
- **[硬伤] README:20 谎称支持 `tushare`**，但 `providers/__init__.py` 无该 adapter（line 33 抛 `Unknown data provider`）。必须删除声明或真正实现 adapter。
- **[好] 无数据泄漏**：`grep shuffle` 在 `src/` 仅出现于 `split.py:3` 的文档字符串（警示 shuffle 错误），无任何 `shuffle=True` 调用。
- **[好] Survivorship 处理**：engine 保留"变暗"标的（`delist_grace_days` 强平而非丢弃），broker 保留 untradeable 权重并按预算重分配 —— 明确避免了隐性幸存者偏差。
- **[好] Sample 数据诚实**：`sample.py:60` 标注 `SYNTHETIC`，README 有 Disclaimer。
- **[缺口] 成本透明度**：仅有 `cost_drag` 单项，未跑"零成本对照"给出 gross vs net Sharpe/CAGR 并排（第 18/19 条建议）。
- **[缺口] README 缺 Limitations / Case Study**：未主动声明 survivorship bias、成本近似、历史≠未来等（第 34/37 条硬性要求）。

### R3 — Senior Software Engineer Review（高级工程）
- **缺 `docker-compose.yml`**：`Dockerfile` 存在但无 compose 一键启动（第 30 条）。
- **docs 缺 mermaid 架构图 + 分模块页**：仅 architecture/quickstart/index 三页（第 32 条）。
- **API 端点命名不符契约**：功能数据可达，但未按第 23 条路径（`/portfolio/*`、`/factors/{name}`、`POST /optimize`、`POST /agent/query`、`POST /backtests`）。
- **测试覆盖不均**：data pipeline/quality、factor engine、costs/broker、attribution 缺专门单测（pipeline_smoke 间接覆盖）。
- **Dead code**：`split.py:73` `dates = ... if False else dates` 无意义三元。
- **命名一致性**：CVXPY 方法名无显式 `alpha-optimization`（可用 mean_variance 替代，但文案应统一）；agent 工具函数名（如 `risk_decomposition`）未对齐第 22 条约定的 `get_*` 命名。

---

## 第三遍：交叉验证与诚实结论

### 前会话"表面完成"的真相（必须正视）
- 前会话结论"13 项任务全部完成、27 passed、API/Dashboard 验证可运行"由**本地完整 `.venv` + 跑通测试**得出。
- 它**未执行第 44 条三轮 Code Review**，也**未逐条对照 50 条**。
- 铁证：代码 push 到 GitHub 后 **CI 第一次真跑就挂**（integration job 缺 matplotlib）—— 本地完整安装掩盖了 CI 缺 extra。
- 本次亲证还有：**engine 成本漏记收益（真实 bug）**、**README 谎称 tushare（诚信硬伤）**、**regime/stress/docker-compose/端点契约/README 章节缺失**。
- 结论："能跑 / 测试过" ≠ "达到 Master Prompt 标准"。

### 代理审查 vs 亲自核实的差异
- 代理指出的 4 个"明显 bug"中，**仅 engine 成本漏记为真**；`bench_cagr 年化` 与 `broker 双重缩放` 是**误报**（已证正确）。其余"部分/缺失"结论成立。

### 按优先级排序的修复清单（诚信 > 正确 > 完整）
- **P0（诚信 + 正确，必须立即修）**
  1. 修 `README.md:20`：删除 tushare 声明，或实现 Tushare adapter。
  2. 修 `engine.py` 成本漏记 returns（step 3 后回写 `rets[date]`）。
- **P1（完整性，影响"可展示"标准）**
  3. `split.py` purge/embargo 去掉静默 ×1.5（或显式 safety factor + 注释）。
  4. 补 `docker-compose.yml`（API + Dashboard 一键起）。
  5. 新增 Market Regime 模块 + dashboard by-regime 展示。
  6. 新增 Stress Testing / scenario shock。
  7. 对齐 API 端点契约（补 `/portfolio/*`、`/factors/{name}`、`POST /optimize`、`POST /agent/query`、`POST /backtests`）。
  8. README 补 Limitations + Case Study 章节；docs 补 mermaid 架构图 + 分模块页。
- **P2（工程质量）**
  9. 补 data/factor/cost/attribution 专门单测。
  10. 清理 `split.py:73` dead code；统一 CVXPY 方法名与 agent 工具命名。
  11. 增加 gross vs net Sharpe/CAGR 对照输出。

---

## 下一步
按 Master Prompt 第 50 条"继续直到达标"的授权，审查完成后应直接进入修复。鉴于本次指令为"过三遍再说"，此处先交付审查结论；待你确认后，从 **P0（README 诚信 + engine 成本 bug）** 起按 P0→P1→P2 顺序修复，每阶段跑 `pytest` + `ruff` + 重新推送 CI。
