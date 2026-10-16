# 目标格式数据契约

本文是清理后数据的唯一目标格式契约，只定义数据内容、表示方式和存储结构。源数据的字段解释、转换方法及人工检查工具分别维护。

## 1. 数据内容

数据按 episode（一次连续操作轨迹）组织，每个 episode 包含：

| 内容 | 记录形式 |
|---|---|
| 左臂、右臂实际 state | 各自的 EEF 绝对位姿 |
| 左、右夹爪实际 state | 各自的开合程度，范围 `[0,1]`；`0` 完全闭合，`1` 完全张开 |
| RGB | 一个或多个相机的彩色视频及逐帧时间戳 |
| instruction | 带起止时间的任务指令文本列表 |

不记录 action、关节状态、速度、深度、触觉、力或音频。

## 2. 时间模型

**同一 episode 的所有数据流共用一个 clock，各流按原生节奏独立、异步记录。** 左右臂位姿、左右夹爪、各相机均有自己的时间戳序列，不要求频率或样本数相同。

- 每个样本携带 `timestamp_ns`：`int64`，单位纳秒，表示相对该 episode 共同时间原点的采样时刻。
- 共同时间原点为本 episode 各采样流（四路 state 与全部相机索引）首样本中的最早者；instruction 不参与取原点。各流不能分别以自己的首样本归零。
- 时间只在 episode 内可比；不记录 clock 标识或绝对时刻，不承诺跨 episode 的时间对齐。
- 保留原生采样时刻与间隔；不插值、重采样或补帧，不通过行号或 FPS 推算时间。
- 每个流内按时间非递减排列。行号只表示该流内的样本顺序，不用于跨流配对。
- instruction 按共享 clock 的起止时间标注，不作为周期采样流。

## 3. State 表示

### 坐标系

左右臂 EEF 位姿各自相对于一个 episode 内**时不变**的参考坐标系：该系相对外部世界保持恒定，不得使用随时间运动的 link（如人形躯干 link）。**不要求左右臂使用同一参考系**（臂间关系不在本契约消费范围内，UMI 等源也无法提供）。除时不变外，不规定其原点与轴向：各源可沿用各自天然的固定系（如机器人 base、odom 系、或 episode 起始工具位姿）。参考系只需在 episode 内时不变；其身份不影响本格式的消费，故不记录。

> 注：下游由 state 导出的 delta 采用 body-frame 定义 `Δ = T_k⁻¹ · T_{k+1}`，参考系的原点与轴向不影响 delta，这是上文不规定其轴向的前提。EEF（body）系的正向已由下文唯一固定，故 delta 的表达不随源变化，可直接跨源比较。

每个 EEF 坐标系原点为对应夹爪的工具中心点（TCP），正向由本契约唯一固定，不随导出变化：

- **+X**：工具接近方向，从工具安装基准（法兰/腕部原点）指向工具工作端（TCP）；工具沿 +X 平动即靠近并接触工件。
- **+Z**：工具固有「上」方向，即掌背（手背）法向；与 +X 正交，由模型或硬件声明。
- **+Y**：`+Y = +Z × +X`。

有了带正负的接近方向与上方向，开合轴的正向由右手性唯一确定，无需区分左右夹指。源须通过一个固定变换把自身原生工具系映射到本规范系；该变换及 TCP 的物理位置记录在源契约中，同一工具在所有 episode、所有源中一致。

pose 表示从 EEF 坐标系到参考坐标系的变换：

```text
p_reference = R @ p_eef + t
```

### 位姿与夹爪字段

左右臂、左右夹爪分别存为独立 Parquet 表，每行一个原生样本：

| 数据流 | 字段 | 类型 | 含义 |
|---|---|---|---|
| 所有 state 流 | `timestamp_ns` | `int64` | 共享 clock 下的采样时刻 |
| 左/右 EEF | `pose` | `float64[12]` | 位置 3 维 + 完整旋转矩阵 9 维 |
| 左/右夹爪 | `openness` | `float32` | 实际开合程度，`[0,1]` |

Parquet 实体的列名与上表一致。`pose` 为单个列，Arrow 类型 `fixed_size_list<float64, 12>`（定长 12，不可空），不拆成多个标量列、不用 struct；`timestamp_ns`、`openness` 均为标量列（`int64`、`float32`）。

`pose` 的固定顺序为：

```text
[x, y, z, R00, R10, R20, R01, R11, R21, R02, R12, R22]
```

`[x,y,z]` 是 TCP 在参考系中的位置，单位米；后 9 维是旋转矩阵 `R` 的三列，按列优先展平。有效旋转须为正交矩阵且行列式为 `+1`。

`openness` 的 `[0,1]` 归一化由源契约负责，导出不做重标定；转换脚本可将有限的越界值裁剪到 `[0,1]`，范围内的值直接透传，非有限值报错。具体源的裁剪规则记录在转换说明中。

## 4. RGB

每个相机独立保存为无音轨的 H.264 MP4，并配套逐帧 Parquet 索引：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_index` | `int64` | 视频按显示顺序解码后的帧序号，从 0 开始 |
| `timestamp_ns` | `int64` | 该帧在共享 clock 下的采集时刻 |

索引与视频帧一一对应。MP4 只承载帧图像，可按固定帧率编码；帧序按原生顺序保留，帧的实际采集时间完全以索引中的 `timestamp_ns` 为准，不由 MP4 的容器时序或帧率推算（可变帧率由此表达）。

`camera_id` 为一个 episode 内唯一的文件名标识：仅小写 ASCII 字母、数字与下划线（`^[a-z0-9_]+$`），不含 `.`、`/`、空格。同一物理相机在该数据集所有 episode 中必须使用同一 `camera_id`；不要求跨数据集同名。取名建议按视角，如 `top`、`left_wrist`、`right_wrist`；立体相机用 `top_left`/`top_right`。

## 5. instruction

`instruction` 在 episode 元数据中保存为列表 `instructions`，每条记录一段生效区间：

| 字段 | 类型 | 含义 |
|---|---|---|
| `start_ns` | `int64` | 该指令生效起点，相对共同时间原点 |
| `end_ns` | `int64` | 该指令生效终点，须大于 `start_ns` |
| `text` | `string` | 非空 UTF-8 任务指令文本 |

至少一条。区间按 `start_ns` 非递减排列且互不重叠。整集仅一句指令时覆盖全轨迹：`start_ns = 0`（共同时间原点），`end_ns` 取该 episode 全部数据流（四路 state 与全部相机索引）中最大的 `timestamp_ns`。

## 6. 存储结构

```text
dataset/
├── episodes.jsonl
└── episodes/<episode_id>/
    ├── state/
    │   ├── left_eef.parquet
    │   ├── right_eef.parquet
    │   ├── left_gripper.parquet
    │   └── right_gripper.parquet
    └── rgb/
        ├── <camera_id>.mp4
        └── <camera_id>.parquet
```

每个 episode 必须含四路 state 与至少一个相机；上表中四张 state Parquet 及每个相机的 MP4/索引均不得缺失。

`episodes.jsonl` 每行一个 episode，记录 `episode_id`、本集相机及指令区间。所有 episode 级信息集中在这一处，不再有 per-episode JSON 文件。各字段的具体定义见第 7 节。

## 7. 元数据 schema

唯一的元数据是逐 episode 的 `episodes.jsonl`。以下为契约字段定义；未列出的字段不属于契约，实现应忽略。所有 `*_ns` 字段均为 `int64` 纳秒。

UTF-8 JSONL，每行一个 episode 对象，行顺序不限：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `episode_id` | string | 是 | 等于 `episodes/<episode_id>/` 目录名 |
| `cameras` | string[] | 是 | 本集包含的相机，非空，须与 `rgb/` 内容一一对应，命名规则见第 4 节 |
| `instructions` | object[] | 是 | 指令区间列表，字段见第 5 节，至少一条 |

全体 `episode_id` 须与 `episodes/` 下的子目录一一对应。

```json
{"episode_id":"episode_0001","cameras":["left_wrist"],"instructions":[{"start_ns":0,"end_ns":12000000000,"text":"pick up the cup"}]}
{"episode_id":"episode_0002","cameras":["left_wrist"],"instructions":[{"start_ns":0,"end_ns":8000000000,"text":"open the drawer"}]}
```
