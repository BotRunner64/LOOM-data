# 导出数据异常检测

入口 [`../../scripts/check/anomaly_detection.py`](../../scripts/check/anomaly_detection.py) 只消费
[目标格式契约](../contract/export-contract.md)，实现 [三条异常规则](../contract/anomaly_detection.md)。

在项目根目录运行：

```bash
python scripts/check/anomaly_detection.py dataset/processed/ABC-130K
# 等价的模块入口；可指定输出位置及阈值
python -m scripts.check.anomaly_detection dataset/processed/ABC-130K \
  --output /tmp/abc-anomalies.jsonl \
  --max-boundary-idle-s 4 \
  --max-internal-idle-s 12
```

默认输出 `<dataset>/anomalies.jsonl`，每行只有 `index`、`stream`、`start_ns`、`end_ns`、`reason`。
`index` 是 `episodes.jsonl` 的物理行号（从 0 开始；跳过空行但仍计入行号）。
没有异常时输出空文件。重复运行替换报告；全量检测成功才原子替换，失败保留之前的报告并返回非零退出码。
已有报告可用 [`anomaly_stats.py`](anomaly-stats.md) 在终端汇总，并显示各异常原因涉及的 episode 数。
输出必须是 `.jsonl`，不能覆盖 `episodes.jsonl` 或写入 `episodes/`。
生成后可用 [导出数据 viewer](export-viewer.md#异常记录浏览) 筛选异常并跳转到对应时间；
viewer 只依赖异常记录格式，不依赖这些检测规则。

逐 episode 读取四路 state 和相机 Parquet 索引，不解码视频。各流保留原生时间戳与采样数；
episode 起点为 0，终点取四路 state 和所有相机索引的最大时间戳，instruction 不参与确定终点。
无效数值、空流、负时间戳、时间倒退或无效旋转矩阵会报错；重复时间戳合法，突变和中段静止规则只保留同一时间戳的最后一个值，与 viewer 一致。
本工具不替代导出格式完整性校验，例如不会校验 MP4 内容。

记录按 episode 行号、规则顺序（首尾静止、突变、中段静止）输出，同一规则按流与时间顺序输出，
中段静止按区间再按流输出。流顺序固定为 `left_eef`、`right_eef`、`left_gripper`、`right_gripper`；
首尾静止若多路并列，取此顺序中的第一路。整段静止且超时仍按契约输出两条首尾记录。
不同规则的记录不去重；同一路突变按事件合并，位置与朝向共同触发只输出一条 `state_jump`。

所有阈值都可通过命令行覆盖：

| 参数 | 默认值 | 单位 / 含义 |
|---|---|---|
| `--position-tolerance-m` | 0.01 | 米，静止、尖峰恢复及尖峰上下文共用 |
| `--orientation-tolerance-deg` | 5 | 度，静止、尖峰恢复及尖峰上下文共用 |
| `--openness-tolerance` | 0.05 | 开合量，静止、尖峰恢复及尖峰上下文共用 |
| `--max-boundary-idle-s` | 3 | 秒 |
| `--position-jump-min-m` | 0.02 | 米，扣除前后样本漂移后的最小尖峰幅度 |
| `--orientation-jump-min-deg` | 10 | 度，扣除前后样本漂移后的最小尖峰幅度 |
| `--openness-jump-min` | 0.25 | 扣除前后样本漂移后的最小尖峰幅度 |
| `--jump-max-duration-s` | 0.05 | 秒，最大允许相邻采样间隔及事件合并间隔 |
| `--jump-context-s` | 0.1 | 秒，每侧稳定上下文窗口 |
| `--max-internal-idle-s` | 10 | 秒 |

四路只保留稳定上下文中的孤立单帧尖峰：一个样本明显偏离前后样本，下一样本恢复，且前后各 100 ms 同一分量稳定。
不再报告持续阶跃、多帧偏移或有明显运动上下文的往返。具体幅度、稳定性和覆盖边界见
[规则 2](../contract/anomaly_detection.md#规则-2state-孤立单帧尖峰state_jump)。

所有参数须有限且非负；最大采样间隔至少 1 ns，稳定窗口不小于其两倍。

调整规则或阈值后需重新运行检测以更新报告；已运行的 viewer 在启动时加载报告，需要重启。
默认值是待人工校准的初始值，没有突变记录不代表所有样本都已验证正常。
使用项目依赖中的 NumPy 与 PyArrow；并行入口使用 `tqdm` 显示进度。

## 并行检测与自动 resume

入口 [`scripts/check/parallel.py`](../../scripts/check/parallel.py) 复用相同的三条规则与阈值，
采用与[并行转换](parallel-conversion.md)相同的 `prepare → worker → finalize` 流程。
`--input-dir`（也可写作 `--dataset-dir`）指向已经导出的数据集，所有阶段都需指定。
默认输出 `<input-dir>/anomalies.jsonl`；使用 `--output` 自定义报告位置时，所有阶段须传入同一路径。
原串行入口保持可用。

本地多进程：

```bash
python -m scripts.check.parallel run \
  --input-dir dataset/export/ABC-130K \
  --workers 8
```

中断后重跑同一命令自动 resume，可以改变 `--workers`。
已完成 episode（包括零异常）直接跳过，不重新读取其 Parquet 内容或计算规则。
阈值只在 `prepare` / `run` 指定，例如追加 `--max-boundary-idle-s 4`；
后续 worker 和 finalize 从计划读取配置。

Slurm 由调用者按顺序执行三阶段，Python 不调用 `srun`：

```bash
# 单进程运行一次
python -m scripts.check.parallel prepare --input-dir dataset/export/ABC-130K

# 由调用者配置 Slurm 资源；每个 task 一个 worker
srun python -m scripts.check.parallel worker --input-dir dataset/export/ABC-130K

# 全部 worker 退出后，单进程运行一次
python -m scripts.check.parallel finalize --input-dir dataset/export/ABC-130K
```

worker 默认读取 `SLURM_PROCID` 与 `SLURM_NTASKS`；本地手动分阶段运行可显式指定
`--worker-id R --worker-count N`。按计划中的任务序号 `ordinal % N == R` 静态分配，
空行不产生任务，但结果 `index` 始终保留 manifest 的物理行号。
所有 worker 均限制 NumPy / BLAS / Arrow 线程池为 1，主要通过 worker 数控制并发。
worker 从 `plan.sqlite` 按需读取自己负责的任务。

各节点须能访问同一仓库、Python 环境、导出数据和工作盘。
**同一报告同时只能有一个作业；重新 prepare 前，上一轮所有 worker 必须已经退出。**
prepare 失败则停止，worker 失败则保留状态；待 worker 全部退出后重跑三阶段即可续跑，
可以改变节点数和 worker 数。

### 恢复状态与报告发布

默认工作目录为 `<output>.anomaly-work`。
仓库内 `dataset/export/<数据集>/anomalies.jsonl` 的状态放在
`dataset/work/<数据集>/anomalies.jsonl.anomaly-work/`，不混入导出数据集或转换恢复目录。

```text
<work>/
├── plan.sqlite                # 任务清单、阈值、输入文件大小/mtime、规则代码校验和
├── records/<index>.json       # 原子提交的 episode 检测结果，含校验和；空 results 也算完成
├── errors/<index>.err         # 最近一次 episode 失败的 traceback
├── logs/worker-<attempt>.err   # worker 启动、当前 episode、退出码或原生崩溃栈
└── published.json             # 已发布报告的 SHA-256 与计划身份
```

prepare 清理中断的临时状态与上一轮 worker 日志，保留正式 records 和 episode 错误，
并打印 `done`、`failed`、`unfinished` 进度。
旧 JSON 恢复目录需在所有 worker 退出后清理，再重新 prepare；已有报告保留。
首次 prepare 先扫描 manifest 统计 episode 总数，再显示准备与恢复核对的百分比进度。
恢复审计复用 records/errors 目录名单，避免对没有记录的 episode 逐个查询记录是否存在；
使用 8 个线程、每批最多 256 集核对输入大小与修改时间，以及已有结果的内容和校验和。
每个输入文件通过一次 stat 获取类型、大小和修改时间；未完成的 episode 也会核对全部输入。
该并发仅用于 prepare/finalize 进程内的文件 I/O，与检测 worker 数无关。
使用 `tqdm` 显示百分比、速度与预计剩余时间；重定向 stderr 或 Slurm 日志中每约 5 秒刷新进度，阶段开始和结束时也输出。
设置 `ROBOT_DATA_HUB_NO_PROGRESS=1` 可关闭进度显示。
worker 按 episode 读取四路 state 和相机 Parquet 索引，检测结果写入临时文件后原子 rename
到 records；这一 rename 是提交点。不解码视频，也不改动原始导出文件。
episode 失败会保存 traceback 并继续其他任务，该 worker 最终返回非零退出码；
下一轮只重试未提交的 episode，成功后清除旧错误。
worker 日志即使成功也会保留，其 `.err` 后缀本身不代表失败；SIGKILL 或节点故障仍需结合 Slurm 日志排查。

finalize 要求所有 episode 都有有效提交，**不省略失败集**。
全部完成后，按 manifest 顺序流式合并，各 episode 内规则和流顺序与串行入口一致，
最终报告可与串行输出逐字节比较。报告只含契约规定的五个字段。
全量成功才原子替换报告；失败或未完成均保留旧报告。
报告 rename 后、发布标记写入前中断，可重新 finalize。
发布后重复运行会核对恢复状态与报告校验和，成功返回，不重新计算检测。

恢复过程中，manifest 及检测读取的 Parquet 文件须保持不变，工具检查大小与修改时间，
不对输入视频或 Parquet 做全量散列。检测规则代码和阈值也须与计划一致；
修改数据、规则或阈值后，使用新的 `--output` 路径建立计划。
损坏的计划、结果记录、计划外记录、输入变化或已发布报告被修改都会报错。
调整阈值后生成新报告，可通过 viewer 的 `--anomalies` 指定该文件（见[viewer 用法](export-viewer.md)）。
