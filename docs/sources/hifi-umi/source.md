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
工具轴向与目标契约一致。raw EE 按**固定于机身的虚拟指尖坐标系**解释，
不把它当作随开合移动的实际指尖中点。该解释来自下述公开说明与真实样本交叉核验，
不是来自尚未公开的上游位姿导出代码。

### 刚体原点核验（2026-11-10）

- [论文 §3.1.1 与 §3.3.2](https://arxiv.org/html/2607.25895)说明，手部轨迹由刚性安装的
  标记块跟踪恢复；编码器独立测量开合角。这支持刚体追踪，但单独不足以证明最终发布的
  指尖位姿未加入开合补偿。
- 实际检查发布版本 `a53b7b5784afdd50b2fda9195c9f724ef75ffdaf` 的
  `chunk-0000/part-0000`：`info.json` / `modality.json` 将 EEF pose 与开合角分列，
  没有提供 marker-to-EE 变换或开合补偿公式。该 shard 共 478,810 行。
- 在第 857 集、第 892–897 帧，右手开角从 `0.0995856971` 增至 `0.4724658728 rad`
  （约 5.7° → 27.1°），raw EE 平移仅 `0.000519126 m`，相关有效性标记均为 true。
  对照 `head_main`、`right_hand_up`、`right_hand_down` 的端点图像，夹指明显张开，
  机身和背景基本稳定，没有足以抵消厘米级指尖运动的机身反向移动。
- 按对称旋转指、12 cm 指长的假设，实际指尖中点应随上述开合变化约 `0.012551595 m`。
  观测与这种运动中点解释不符，支持将发布的 raw EE 作为机身固定虚拟坐标系。

原始六帧状态、数据文件 SHA-256、版本、视频定位时间与核验图像哈希保存在
`tests/fixtures/hifi_umi_rigid_frame_samples.json`。这是一项有来源和样本依据的工程判定；
不宣称已经取得作者对所有批次的变换源码或逐套硬件标定。

### 默认后端基准

采用沿虚拟指尖系 `-X` 后退 12 cm 的**名义虚拟安装点**：

```text
p_raw_mount = [-0.12, 0, 0] m
p_world_mount = p_world_raw + R_world_raw @ p_raw_mount
R_world_mount = R_world_raw
```

12 cm 依据下文维护者给出的指尖到转轴近似长度；这里将它作为名义后端/转轴基准，
不把它声称为已精确测量的安装接口。左右手、所有 episode 使用同一固定默认值，
不按每帧开角使用 `cos(angle)` 修正，也不改变源旋转。
`--left-eef-offset X Y Z` / `--right-eef-offset X Y Z` 可分别覆盖安装点在 raw EE 系中的
坐标（米）；只覆盖一手时另一手保留默认值。获得更准确的后端几何后应更新偏移并重导。

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
