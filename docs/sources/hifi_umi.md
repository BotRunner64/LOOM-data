# HiFi-UMI-2K

| | |
|---|---|
| 上游 | [simple-world-lab/HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K) |
| 数据集键 | `hifi_umi` |
| 位姿来源 | 源记录的手部位姿（虚拟指尖系），平移到名义后端安装点 |
| 机器人模型 | 不需要 |
| 不合格的 episode | 报错，记为失败 |

```bash
loom-data download hifi_umi select --count 5
loom-data download hifi_umi download
loom-data convert run --dataset hifi_umi --workers 8
```

本数据源的选项（均可省略）：

| 选项 | 含义 | 默认 |
|---|---|---|
| `--left-eef-offset X Y Z` | 左手安装点在 raw EE 系中的坐标，米 | `-0.12 0 0` |
| `--right-eef-offset X Y Z` | 右手安装点在 raw EE 系中的坐标，米 | `-0.12 0 0` |
| `--gripper-open-rad` | 全开时的单指角，弧度 | 35° |
| `--gripper-closed-rad` | 闭合时的角度，弧度 | `0` |

通用选项、并行与续跑见[使用说明](../usage.md)；输出格式见[导出契约](../contract/export-contract.md)。

## 原始数据

依据随数据下载的 `source/info.json`、`source/modality.json`，以及官方
[数据集说明](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)。

### 文件组织与字段

原始发布按 `chunk-XXXX/part-YYYY/` shard 组织，每个 shard 是独立的 LeRobot v3 数据集；
episode 编号只在 shard 内唯一。转换器可直接读取完整本地数据集：每个 shard 包含
`meta/{info.json,modality.json,tasks.parquet}`、`meta/episodes/**/*.parquet`、
`data/chunk-*/file-*.parquet` 与 `videos/<video_key>/chunk-*/file-*.mp4`。
episode 元数据记录逐帧表的文件索引、全局行号区间，以及每路拼接视频的文件索引与时间区间。

`loom-data download hifi_umi` 选择一个 shard，
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

### 坐标系与夹爪

官方[坐标系说明](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K#-coordinate-frames)
声明：两手原点在指尖，+X 沿指尖指向前方，+Y 向左，+Z 向上；手部轨迹表达在
同一固定 world 系中，而非运动中的头部相机系。world 原点任意，不跨 recording 对齐。
工具轴向与目标契约一致。raw EE 按**固定于机身的虚拟指尖坐标系**解释，
不把它当作随开合移动的实际指尖中点。该解释来自下述公开说明与真实样本交叉核验，
不是来自尚未公开的上游位姿导出代码。

#### 刚体原点核验（2026-11-10）

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

#### 默认后端基准

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

### 时间和相机

`timestamp` 为 episode 相对秒数（float32）。公开 LeRobot 表已经对齐到每行一个
视频帧，四路 state 与六路相机在这个发布格式中共用该行时间戳；原始传感器异步
时间戳并未包含在这些列中。下载器保留表内时间数值，切出的视频按显示顺序与表行
一一对应；`frame_index` 从零连续，`index` 为 shard 内全局行号。

六路视频键为 `observation.images.` 加上 `head_main`、`head_main_stereo_right`、
`left_hand_up`、`left_hand_down`、`right_hand_up`、`right_hand_down`。
下载器输出已经是无音轨 H.264 MP4，帧数等于该 episode 的 `length`。
源 MP4 的容器时间和 episode 元数据的 `from/to_timestamp` 用于切片定位，不能
替代采样表中的时间戳。

## 转换规则

### 输入布局

`--input-dir` 可以指向包含 `chunk-XXXX/part-YYYY/` 的完整下载根目录，也可以指向
其中一个 `part-YYYY/`，或包含 `source/` 元数据与 `episodes/` 的子集目录。不直接读取远端数据。
完整分片读取各自的 `meta/info.json`、`meta/modality.json`、`meta/tasks.parquet` 和
`meta/episodes/**/*.parquet`，按元数据中的文件索引与数据范围读取逐帧表，并按六路
视频各自的 `[from_timestamp, to_timestamp)` 截取拼接视频。从前一个关键帧开始解码，
仅保留当前 episode 的显示帧，重新编码为 H.264（CRF 18、fast），校验时间区间与帧数。
MP4 播放时间从零开始，输出 Parquet 时间戳仍来自原生 `timestamp`，不从视频 FPS 推算。
完整数据输出 ID 为 `chunk_XXXX_part_YYYY_episode_NNNNNN`，避免分片内重复编号；子集沿用 `episode_NNNNNN`。
完整数据按 shard 分配 worker：每个 worker 只缓存当前 shard 的一个数据文件所需列，六路视频逐路处理。

### 位姿和夹爪

- 只读 `observation.state`：右 EEF `0:9`、右夹爪 `9`、左 EEF `10:19`、左夹爪 `19`。
- 保留米制单位、固定 world 参考系和原工具轴向；按 `p_mount = p_raw + R @ offset`
  将机身固定的虚拟指尖原点移至名义后端安装点。左右手默认偏移均为 `[-0.12,0,0] m`，
  对应沿工具 `-X` 后退 12 cm；不依赖开合角，不使用 `cos(angle)` 动态修正。
  依据和适用限制见上文「坐标系与夹爪」。
- 从原始旋转前两行恢复第三行，再按契约写出完整矩阵；不是将源 6D 数值直接透传，
  也不是将整个旋转取逆。先用绝对容差 `1e-5` 检查行正交与单位长度，再用
  Gram–Schmidt 消除 float32 舍入误差，输出位置加按列展平的旋转矩阵 `float64[12]`。
- 夹爪按单指角处理，全开值固定采用 35°（约 `0.6108652382` rad），闭合值为 0，
  直接以 `clip(angle / radians(35), 0, 1)` 归一化，不乘以 2。
  两手及全部 episode 使用相同端点。覆盖端点时公式为
  `clip((angle - closed_rad) / (open_rad - closed_rad), 0, 1)`。
  有限越界值裁剪，非有限值报错；不按 episode 或 shard 的统计极值重标定。

### 时间、视频和指令

- 使用 `round(float(timestamp) * 1e9)` 转为 int64 纳秒；保留源 float32 的精度及
  间隔，不根据 FPS 或行号重新生成时间。全部流统一减去首行时间。
- 输出四张独立 state 表。源发布本身是同步行格式，转换不会虚构异步采样时刻。
- 相机 ID 为去掉 `observation.images.` 后的固定名称。每个视频必须是仅含一个
  H.264 视频流的 MP4；逐帧解码验证显示帧数和分辨率后逐字节复制，不再次有损编码。
  帧索引为 `0..N-1`，其时间来自源表；视频容器 PTS 不参与采样时刻计算。
- 按逐行 `task_index` 的变化生成指令区间。第一条从共同零点开始，末段终点为最后
  样本时间；同一时刻最后一个任务生效，零时长区间省略，相同文本连续行合并。

### 校验

检查源 embodiment/布局、episode 身份和长度、连续帧号/全局行号、任务映射、全部
六路相机和视频帧数。空轨迹、零时长、时间倒退、缺失文件、非有限值、无效旋转均报错。
任何 `observation.state_valid` 或 `valid.frame` 非 true 的 episode 都拒绝转换（记为失败）；当前实现
不删除、补齐或插值这些样本，避免丢弃有效性标记后将前值填充/黑帧误作正常观测。
