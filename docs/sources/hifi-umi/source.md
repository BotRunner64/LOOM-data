# HiFi-UMI-2K 源格式说明

依据随数据下载的 `source/info.json`、`source/modality.json`，以及官方
[数据集说明](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)。

## 文件组织与字段

原始发布按 `chunk-XXXX/part-YYYY/` shard 组织，每个 shard 是独立的 LeRobot v3 数据集；
episode 编号只在 shard 内唯一。转换器可直接读取完整本地数据集：每个 shard 包含
`meta/{info.json,modality.json,tasks.parquet}`、`meta/episodes/**/*.parquet`、
`data/chunk-*/file-*.parquet` 与 `videos/<video_key>/chunk-*/file-*.mp4`。
episode 元数据记录逐帧表的文件索引、全局行号区间，以及每路拼接视频的文件索引与时间区间。

本项目下载器 `scripts/download/hifi_umi_subset.py` 选择一个 shard，
将其中选定的 episode 拆为：

```text
dataset/raw/HiFi-UMI-2K/
├── source/{info.json,modality.json,tasks.parquet,...}
└── episodes/episode_NNNNNN/
    ├── episode.json
    ├── data.parquet
    └── videos/observation.images.<camera>.mp4
```

`data.parquet` 的 `observation.state` 为 float32[20]，右手在前、左手在后。
每手 10 维为位置 xyz（米）、旋转矩阵前两行按行展开的 6 维、夹爪角度（弧度）。
`observation.state_valid` 是逐维 bool[20] 掩码；源导出器对无效 state 沿用前值并
置掩码为 false。`valid.frame` 是整帧有效标记，无效视频索引可能已经被源导出器
替换成黑帧。无效行仍保留在表和视频中。

`action` 是绝对的下一状态目标，末行重复最后目标，不能当作当前实测 state。
`task_index` 关联 `source/tasks.parquet` 的 `task_index` / `task`；episode 元数据
中的 `tasks` 只列出任务文本，逐行归属以 `task_index` 为准。

## 坐标系与夹爪

官方[坐标系说明](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K#-coordinate-frames)
声明：两手原点在指尖，+X 沿指尖指向前方，+Y 向左，+Z 向上；手部轨迹表达在
同一固定 world 系中，而非运动中的头部相机系。world 原点任意，不跨 recording 对齐。
本项目将源指尖 frame 作为 TCP；工具轴向与目标契约一致，固定变换为单位阵，
平移为零，不额外增加腕部或指长偏移。

夹爪字段只声明为 opening angle（rad），未提供归一化标定。
维护者在[硬件尺寸回复](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/discussions/2)
给出的约 35° 单指 / 70° 双指总开角是近似机械范围，未明确表内标量对应哪种角度。
本地整个 shard 的有效夹爪角度最大为右手 29.62°、左手 29.27°，结合视频检查，
本项目确定按单指角解释该字段：闭合 0 rad，全开 35°（`35 * pi / 180` rad）。
归一化直接除以该全开值，不将记录角度乘以 2，不使用 70° 作为分母。
这一字段解释是本项目依据数据确定的约定；每条轨迹的观测极值不作为机械端点。

## 时间和相机

`timestamp` 为 episode 相对秒数（float32）。公开 LeRobot 表已经对齐到每行一个
视频帧，四路 state 与六路相机在这个发布格式中共用该行时间戳；原始传感器异步
时间戳并未包含在这些列中。下载器保留表内时间数值，切出的视频按显示顺序与表行
一一对应；`frame_index` 从零连续，`index` 为 shard 内全局行号。

六路视频键为 `observation.images.` 加上 `head_main`、`head_main_stereo_right`、
`left_hand_up`、`left_hand_down`、`right_hand_up`、`right_hand_down`。
下载器输出已经是无音轨 H.264 MP4，帧数等于该 episode 的 `length`。
源 MP4 的容器时间和 episode 元数据的 `from/to_timestamp` 用于切片定位，不能
替代采样表中的时间戳。
