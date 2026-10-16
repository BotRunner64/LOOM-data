# 异常结果快速查看

入口 [`../../scripts/check/anomaly_stats.py`](../../scripts/check/anomaly_stats.py) 读取已有
`episodes.jsonl` 和 `anomalies.jsonl`，在终端汇总异常结果。只使用 Python 标准库，
不重新检测、不遍历 episode 目录、不读取 Parquet 或视频，也不修改数据。

在项目根目录运行：

```bash
# 默认汇总 dataset/export 下的直接子数据集
python scripts/check/anomaly_stats.py

# 一个数据集、多个数据集或它们的父目录；重叠路径会去重
python scripts/check/anomaly_stats.py dataset/export/ABC-130K
python -m scripts.check.anomaly_stats dataset/export

# 自定义报告位置，只能对应一个数据集
python scripts/check/anomaly_stats.py dataset/export/ABC-130K --anomalies /tmp/abc-anomalies.jsonl

# JSON 汇总
python scripts/check/anomaly_stats.py dataset/export --json > /tmp/anomaly-stats.json
```

文本只输出一张总览表，每个数据集一行，最后是合计：

- `Episodes`：episode 总数。
- `Records`：异常记录数。
- `Affected`：涉及异常的 episode 数。
- `Rate`：涉及异常的 episode 占比。
- `boundary_idle`、`internal_idle`、`state_jump`：分别涉及首尾静止、中段静止、单帧尖峰的 episode 数。
  报告若出现其他原因，自动追加对应列。

数字使用千位分隔符并靠右对齐，表头和合计前加分隔线。

统计口径：

- **Records** 是报告行数，不是去重后的物理事件数。`internal_idle` 同一区间会按四路各记一条。
- **Affected 和原因列** 按数据集内的 manifest index 去重；同一 episode 的同类异常只计一次。
  各原因的 episode 数可能重叠，不能直接相加。
- **Rate** 是涉及异常的 episode 数除以该数据集全部 episode 数。
  合计分母只包含存在报告的数据集；空清单时占比为 `—`（JSON 为 `null`）。
- 报告缺失时表下列出对应数据集，相关计数为 `—`（JSON 为 `null`，`report_status` 为 `missing`），不会当作零异常。
  存在但为空的报告表示零记录。全部报告缺失时合计记录数为 0，但占比为 `—`。
  自定义 `--anomalies` 文件不存在则报错。
- `index` 使用清单从 0 开始的物理行号，空行也占行号。脚本验证清单 ID 的唯一性、
  报告字段和索引有效性；不根据 instruction 时间限制异常区间，也不验证报告与当前数据是否同步。

JSON 包含 `datasets` 和 `total`。
各数据集包含 `report_status`、`episode_count`、`record_count`、`affected_episode_count`、
`affected_episode_rate`（0 到 1 的比例）、`by_reason`，以及数据集与报告路径。
`by_reason` 按原因提供记录数、去重后的 episode 数和占比；没有记录的原因不写入该映射。
`total` 还提供 `dataset_count`、`report_count`、`missing_report_count`、`reported_episode_count`。
报告逐行处理；内存用于 episode ID 映射和按原因的去重集合。
所有输入验证成功后才输出报告；损坏记录报出文件和行号，返回非零退出码。
Ctrl+C 返回 130。

要结合视频和曲线逐条复核，可启动现有 [viewer](export-viewer.md#异常记录浏览)：

```bash
python scripts/viewer/export_viewer.py dataset/export/ABC-130K
```

viewer 自动读取该数据集下的 `anomalies.jsonl`，支持筛选、上一条/下一条和跳转至异常时间。
异常检测与报告生成方式见 [异常检测说明](anomaly-detection.md)。
