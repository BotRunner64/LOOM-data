# HiFi-UMI-2K 转换

入口：`scripts/convert/hifi_umi.py`。读取本地完整 LeRobot v3 分片或本项目下载器拆出的 episode 子集，输出
遵循[目标契约](../../contract/export-contract.md)。源字段依据见[源格式说明](source.md)。

```bash
pip install -e .
python scripts/convert/hifi_umi.py
# 模块入口等价；可先转换一条，输出目录须不存在。
python -m scripts.convert.hifi_umi \
  --limit 1 --output-dir dataset/processed/HiFi-UMI-2K-preview
```

默认输入 `dataset/raw/HiFi-UMI-2K`，默认输出 `dataset/processed/HiFi-UMI-2K`。
`--input-dir` 可以指向包含 `chunk-XXXX/part-YYYY/` 的完整下载根目录，也可以指向
其中一个 `part-YYYY/`，或包含 `source/` 元数据与 `episodes/` 的子集目录。不直接读取远端数据。
完整分片读取各自的 `meta/info.json`、`meta/modality.json`、`meta/tasks.parquet` 和
`meta/episodes/**/*.parquet`，按元数据中的文件索引与数据范围读取逐帧表，并按六路
视频各自的 `[from_timestamp, to_timestamp)` 截取拼接视频。从前一个关键帧开始解码，
仅保留当前 episode 的显示帧，重新编码为 H.264（CRF 18、fast），校验时间区间与帧数。
MP4 播放时间从零开始，输出 Parquet 时间戳仍来自原生 `timestamp`，不从视频 FPS 推算。
完整数据输出 ID 为 `chunk_XXXX_part_YYYY_episode_NNNNNN`，避免分片内重复编号；子集沿用 `episode_NNNNNN`。
运行需要 NumPy、PyArrow、PyAV，
无需机器人模型、GPU 或额外 ffmpeg 命令行程序。

服务器上的完整数据集建议使用[并行转换入口](../../tools/parallel-conversion.md)。按 shard
分配 worker，逐 episode 提交与恢复；每个 worker 只缓存当前 shard 的一个数据文件所需列，
六路视频逐路处理，避免为每集重复读取整个 shard 的 Parquet。

## 位姿和夹爪

- 只读 `observation.state`：右 EEF `0:9`、右夹爪 `9`、左 EEF `10:19`、左夹爪 `19`。
- 保留米制位置和固定 world 参考系。源指尖 TCP 及工具轴向沿用源契约，工具变换为单位阵。
- 从原始旋转前两行恢复第三行，再按契约写出完整矩阵；不是将源 6D 数值直接透传，
  也不是将整个旋转取逆。先用绝对容差 `1e-5` 检查行正交与单位长度，再用
  Gram–Schmidt 消除 float32 舍入误差，输出位置加按列展平的旋转矩阵 `float64[12]`。
- 夹爪按单指角处理，全开值固定采用 35°（约 `0.6108652382` rad），闭合值为 0，
  直接以 `clip(angle / radians(35), 0, 1)` 归一化，不乘以 2。
  两手及全部 episode 使用相同端点；CLI 无需传入夹爪参数。
  如需显式覆盖，保留 `--gripper-open-rad` / `--gripper-closed-rad`，
  要求两者有限且全开值大于闭合值，通用公式为
  `clip((angle - closed_rad) / (open_rad - closed_rad), 0, 1)`。
  有限越界值裁剪，非有限值报错；不按 episode 或 shard 的统计极值重标定。
  单指角约定依据及官方机械范围见源格式说明。

## 时间、视频和指令

- 使用 `round(float(timestamp) * 1e9)` 转为 int64 纳秒；保留源 float32 的精度及
  间隔，不根据 FPS 或行号重新生成时间。全部流统一减去首行时间。
- 输出四张独立 state 表。源发布本身是同步行格式，转换不会虚构异步采样时刻。
- 相机 ID 为去掉 `observation.images.` 后的固定名称。每个视频必须是仅含一个
  H.264 视频流的 MP4；逐帧解码验证显示帧数和分辨率后逐字节复制，不再次有损编码。
  帧索引为 `0..N-1`，其时间来自源表；视频容器 PTS 不参与采样时刻计算。
- 按逐行 `task_index` 的变化生成指令区间。第一条从共同零点开始，末段终点为最后
  样本时间；同一时刻最后一个任务生效，零时长区间省略，相同文本连续行合并。

## 校验和发布

检查源 embodiment/布局、episode 身份和长度、连续帧号/全局行号、任务映射、全部
六路相机和视频帧数。空轨迹、零时长、时间倒退、缺失文件、非有限值、无效旋转均报错。
任何 `observation.state_valid` 或 `valid.frame` 非 true 都拒绝整个转换；当前实现
不删除、补齐或插值这些样本，避免丢弃有效性标记后将前值填充/黑帧误作正常观测。

数据先写入输出目录旁的暂存目录，全部成功才发布；失败自动清理暂存，不覆盖已有输出。
需要容纳整份输出的可用空间。Parquet 写出、秒转纳秒和合规视频复制复用
`scripts/convert/export_common.py`。

测试入口：`python -m unittest discover -s tests -p 'test_convert_hifi_umi.py'`。
端到端测试使用默认 35° 全开值，同时检查 17.5° 映射为 0.5。
