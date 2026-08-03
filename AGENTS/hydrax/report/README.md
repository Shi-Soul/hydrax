# Hydrax 批量实验归档

本目录备份 `/home/bai/MPC-RL/hydrax/examples/to` 中现存的 Hydrax 批量实验、
报告、dashboard、图表、视频和对应代码版本。原始工作目录未被移动或删除。

- 归档时间：`2026-08-03T13:41:39+08:00`
- 归档前代码基线：`76da3a4`
- 完整性清单：`MANIFEST.sha256`
- 结构化元数据：`ARCHIVE_METADATA.json`
- 条目统计：`INVENTORY.tsv`

## 条目索引

| 条目 | 实验 | 数据完整性 | 主要入口 |
|---|---|---|---|
| `01_legacy_open_loop` | 5-task one-shot open-loop tuning 与 evaluation | 完整：160 条 tuning、150 条 evaluation | [说明](01_legacy_open_loop/README.md) / [报告](01_legacy_open_loop/results/report.md) |
| `02_legacy_receding_horizon` | 历史 5-task receding-horizon dashboard | 部分：保留 aggregate 和 30 个视频，逐-seed CSV 缺失 | [说明](02_legacy_receding_horizon/README.md) / [Dashboard](02_legacy_receding_horizon/index.html) |
| `03_cart_pole_horizon_2s` | Cart Pole 2 秒 horizon 重跑 | 完整：30 条 evaluation | [说明](03_cart_pole_horizon_2s/README.md) / [报告](03_cart_pole_horizon_2s/results/report.md) / [Dashboard](03_cart_pole_horizon_2s/index.html) |
| `04_double_cart_pole_scaling` | Double Cart Pole operation-budget scaling | 完整：750 条 episode、150 个 aggregate、30 个最优配置 | [说明](04_double_cart_pole_scaling/README.md) / [报告](04_double_cart_pole_scaling/report.md) / [Dashboard](04_double_cart_pole_scaling/index.html) |

## 目录约定

```text
AGENTS/hydrax/report/
├── 01_legacy_open_loop/
├── 02_legacy_receding_horizon/
├── 03_cart_pole_horizon_2s/
├── 04_double_cart_pole_scaling/
├── ARCHIVE_METADATA.json
├── INVENTORY.tsv
├── MANIFEST.sha256
└── README.md
```

每个条目的 `source/` 保存生成该批次时对应的 Git 代码快照。历史 source 中的
Hydra 路径仍指向 `examples/to/`，用于版本追溯；需要复现实验时，应先在原仓库
恢复对应 commit，而不是直接从归档子目录运行。

## Dashboard 查看

dashboard 包含 JavaScript 和视频相对路径，建议从仓库根目录启动静态服务器：

```bash
python3 -m http.server 8890 --bind 127.0.0.1 --directory AGENTS/hydrax/report
```

然后访问：

- `http://127.0.0.1:8890/02_legacy_receding_horizon/index.html`
- `http://127.0.0.1:8890/03_cart_pole_horizon_2s/index.html`
- `http://127.0.0.1:8890/04_double_cart_pole_scaling/index.html`

## 完整性验证

在本目录执行：

```bash
sha256sum -c MANIFEST.sha256
```

清单覆盖除 `MANIFEST.sha256` 自身以外的所有归档文件。任何原始 CSV、报告、
代码、图片或视频发生变化都会导致校验失败。

## 重要区分

`01_legacy_open_loop/results/benchmark_results.csv` 与
`02_legacy_receding_horizon/results/browser_data.js` 来自不同实验协议，字段和 cost
语义不同。后者缺少逐-seed CSV，不能用前者替代。详细限制记录在各条目的
README 中。
