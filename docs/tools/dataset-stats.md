# 导出数据集快速汇总

入口 [`../../scripts/check/dataset_stats.py`](../../scripts/check/dataset_stats.py) 只读取
[目标格式契约](../contract/export-contract.md) 的 `episodes.jsonl`，输出各数据集与合计的 episode 数量和时间，备注说明时间来自 instruction。

在项目根目录运行：

```bash
# 汇总 dataset/export 下所有直接子数据集
python scripts/check/dataset_stats.py dataset/export

# 默认位置仍为 dataset/processed
python scripts/check/dataset_stats.py

# 单个数据集，或多个数据集路径
python scripts/check/dataset_stats.py dataset/export/ABC-130K
python -m scripts.check.dataset_stats dataset/export/ABC-130K dataset/export/HiFi-UMI-2K

# JSON 报告写入 stdout
python scripts/check/dataset_stats.py dataset/export --json > /tmp/dataset-stats.json
```

输入可以是包含 `episodes.jsonl` 的数据集根目录，或直接包含这些数据集的父目录；不递归发现更深层数据集。
重叠输入路径会去重。只读取清单，不遍历 `episodes/`，不打开 Parquet 或 MP4，不统计磁盘占用，也不提供完整扫描选项。
脚本仅使用 Python 标准库，不依赖 PyArrow。

统计口径：

- **Episode 数量**：清单的非空行数；拒绝重复或不安全的 episode ID。空清单计为 0。
- **时间（Time）**：来自 instruction，每个 episode 取 `instructions` 中最大的 `end_ns`，然后累加。
  指令区间必须是有效的整数纳秒，按时间排列且互不重叠。不会逐段相加，也不会重复累计相机时长。
  当指令覆盖完整轨迹时，这个值就是轨迹总时长；只覆盖部分轨迹时，它仅反映指令结束位置，不验证数据流的真实终点。
  文本显示 `HH:MM:SS`，小时可超过 24；先累加纳秒再截断到整秒。

JSON 包含 `datasets` 和 `total`。每个数据集有 `dataset`、`path`、`episode_count`、`duration_ns`、
`duration_seconds`、`duration_source`；合计包含同样的统计字段。`duration_source` 固定为 `instructions`，
`duration_ns` 是整数纳秒，`duration_seconds` 是截断小数后的整数秒；不再包含 `disk_usage_bytes`。

正常运行只输出最终报告，不输出读取进度日志。文本报告在表格后附备注，说明时间来自 instruction。
记录逐行处理，不保留整个清单；内存主要用于 episode ID 去重集合。
报告在所有数据集成功汇总后输出；清单读取失败或统计字段无效时返回非零退出码。
按 Ctrl+C 可安全中止，退出码为 130。脚本不修改数据集，也不保存断点。

```bash
python -m unittest discover -s tests -p 'test_dataset_stats.py'
```
