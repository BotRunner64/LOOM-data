# MolmoAct2-BimanualYAM

| | |
|---|---|
| 上游 | [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) |
| 数据集键 | `molmoact2` |
| 位姿来源 | 实测关节角经 YAM 模型 FK 计算，左右臂各以自身基座为参考系 |
| 机器人模型 | `assets/robot_models/yam` |
| 不合格的 episode | 报错，记为失败 |

```bash
loom-data download molmoact2 select --count 5
loom-data download molmoact2 download
loom-data convert run --dataset molmoact2 --workers 8
```

通用选项、并行与续跑见[使用说明](../usage.md)；输出格式见[导出契约](../contract/export-contract.md)。

## 原始数据

依据[官方数据卡](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset)、
下载保留的 `source/meta/info.json`、任务表及逐帧 Parquet。

### 文件组织

合并后的 LeRobot v3.0 数据集，`robot_type = bi_yam_follower`。
原仓库的 `data/chunk-*/file-*.parquet` 和各相机 MP4 可容纳多条 episode。
`meta/episodes/` 记录 episode 的数据索引范围及各视频内的片段边界。
转换器可直接读取这个完整本地布局，无需先用下载器拆分 episode。

`loom-data download molmoact2` 的下载产物为：

```text
MolmoAct2-BimanualYAM/
├── source/meta/{info.json,tasks.parquet,tasks_annotated.parquet,...}
├── source/data/chunk-*/file-*.parquet
├── subset_manifest.json
└── episodes/episode_XXXXXX/
    ├── data.parquet
    ├── episode.json
    └── videos/observation.images.{top,left,right}.mp4
```

下载器按 `episode_index` 过滤数据表，按视频片段边界截取并编码成 H.264；
本地每个视频的显示帧顺序对应本 episode 的 `frame_index`。

### 状态与夹爪

`observation.state` 为 `float32[14]`，按 `info.json` 的特征名称确定顺序：

| 切片（Python） | 含义 |
|---|---|
| `[0:6]` | 左臂 `left_joint_0.pos` 至 `left_joint_5.pos`，基座到腕部六关节，rad |
| `[6]` | 左夹爪 `left_gripper.pos`，实际归一化开合程度 |
| `[7:13]` | 右臂 `right_joint_0.pos` 至 `right_joint_5.pos`，基座到腕部六关节，rad |
| `[13]` | 右夹爪 `right_gripper.pos`，实际归一化开合程度 |

`action` 是另一个同维度字段，不是实测状态。源表没有可直接使用的 EEF pose。
官方 [YAM 驱动](https://github.com/allenai/molmoact2/blob/main/examples/yam/gello_min/yam.py)
通过 I2RT `get_joint_pos()` 读取六关节及夹爪；I2RT 在构造实测 state 时
使用 [JointMapper](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robots/utils.py)
把夹爪标定端点归一化到 `0` 闭合、`1` 张开。
这一数值不是米、角度或百分数，不需要再除以夹爪行程或 100。

### 时间和相机

- `timestamp`：逐行 float32 秒，episode 内的 LeRobot 记录时间。
- `frame_index`：episode 内从 0 开始的连续帧序号；`index` 为合并数据集全局行号。
- 每行包含一次双臂状态记录及三相机的对应图像；发布版本仅提供这一条共同时间序列，
  没有各关节、夹爪或相机独立的硬件采样时间戳。
- 本地五条样本的时间值与 float32 的 `frame_index / 30` 一致。
  不能据此宣称恢复了传感器真实采样抖动或原始异步时序。官方另附的
  [LeRobot 转换示例](https://github.com/allenai/molmoact2/blob/main/examples/yam/lerobot_convert.py)
  也没有向 `add_frame` 传入硬件采样时间；该示例不等同于已证明合并数据的全部生成历史。
- 三路 `observation.images.top/left/right` 分别为顶部、左腕、右腕视角。
  源 `info.json` 标为 640×360、AV1；本项目下载器的 episode 视频已经是 H.264。
- `videos/.../from_timestamp`、`to_timestamp` 是各自拼接 MP4 中的定位边界，
  不同相机可以差很大，不是传感器之间的采样时间差；不能加到导出时间上。

### 语言

优先通过 `meta/tasks_annotated.parquet` 的 `episode_index → task` 获取整集详细标注。
官方数据卡要求在缺少有效标注时回退到逐行 `task_index` 查
`meta/tasks.parquet`；任务文本存于该表的 Pandas 索引列 `__index_level_0__`。
下载器还会将任务和标注缓存进 `episode.json`。

## 转换规则

### 输入布局

`--input-dir` 可以指向原始数据根目录（包含 `meta/`、`data/`、`videos/`），
或下载器拆好的子集目录（包含 `source/meta/` 和 `episodes/`）；
完整数据从 `meta/episodes/**/*.parquet` 枚举 episode，按元数据中的 chunk/file 索引
定位帧表与三路拼接 AV1 视频，按各相机自己的时间区间解码并转成单集 H.264（CRF 18、fast）。
数据文件只缓存所需列，最多保留当前一个文件；视频逐路处理。完整数据按 episode 分配 worker。

### YAM 工具几何约定

固定安装的双臂可分别以自身 `arm_{side}_base_link` 为时不变参考系。
仓库 YAM 模型的双臂外参来自 ABC 仿真布局，不代表 MolmoAct2 的安装标定。

沿用同一 YAM 工具的安装基准：EEF 原点是 `link6` 原点，与模型 `tcp` site 的位置重合，
不再使用 `grasp` site 的 `0.1347 m` 指尖偏移；规范工具轴为 `+X = +Z_link6`（接近方向）、
`+Z = +Y_link6`（相机侧掌背法向）、`+Y = +X_link6`。
从模型 tcp site 到规范工具系的固定变换无平移，旋转为：

```text
[ 0  0 -1 ]
[ 0  1  0 ]
[ 1  0  0 ]
```

模型依据和 site 定义见 [ABC-130K](abc130k.md)。
此处采用名义模型几何和关节定义，不代表已逐条标定真实机器人零位、工具长度或安装外参。

### 映射

- 仅使用 `observation.state`；不读取 action，也不输出关节、速度等额外数据。
- 六个实测关节角按 rad 输入 YAM FK；左右臂分别以自身固定基座为参考系，
  不套用 ABC 的双臂安装外参。安装接口原点和工具轴向与 ABC 共用同一实现，
  按上面的固定变换生成位置加按列展平的完整旋转矩阵。
- 夹爪已有归一化：有限值裁剪到 `[0,1]`，范围内透传；NaN、无穷时报错。
- 直接将表内 `timestamp` 四舍五入成 int64 纳秒，减去共同首时刻。
  四路 state 和三相机索引沿用同一记录时间序列，不插值、补帧、排序或按 FPS 重建。
  **源仅保留 LeRobot 记录时间，无法恢复硬件采集时刻及各传感器异步节奏**；
  输出保留源所能提供的时间精度，不能视作独立硬件时间戳的测量结果。
- `top → top`、`left → left_wrist`、`right → right_wrist`。
  子集 MP4 必须为单路无音轨 H.264，逐帧解码确认分辨率稳定、帧数与表行数一致后按字节复制。
  原始拼接视频从前一个关键帧开始解码，仅保留 `[from_timestamp, to_timestamp)` 中的帧，
  校验区间长度、帧连续性和帧数后重新编码为 H.264，无需 GPU。
  容器 PTS 用于定位片段；输出采集时间仍使用帧表的原生 `timestamp`，不加拼接片段偏移。
- 逐 episode 有效详细标注优先，覆盖 `[0, 最后记录时刻]`；缺失、null、空白标注回退到
  标准任务表，按逐行 `task_index` 的文本变化生成不重叠区间。
  相同时间最后一次变化生效，末采样时刻的零长度区间不写出。
  以完整数据的 `meta/` 或子集的 `source/meta/` 任务表为准，不使用 `episode.json.task` 缓存覆盖它。

### 校验

校验本体、版本、14 维状态字段名称及顺序、三路视频特征；
校验 episode 目录 ID、元数据长度、全局索引区间、逐行 episode ID、连续帧序和索引。
时间非有限、倒序、负值或整集零时长，状态维度错误、非有限数，任务索引未知、
可导出元数据声明的一至三路目标视角。任务表重复 ID、没有目标视角、声明的视频文件缺失或帧数不符均报错。

完整数据的 `prepare` 只建立计划；视频区间与帧数的不一致在对应 episode 转换时
报错，其他 episode 可继续提交，修复问题后可恢复。服务器 v1 的 episode 31749
帧表有 1,619 行，但 top/left 视频区间对应 2,832 帧、right 对应 2,658 帧；
转换器会明确拒绝这集，不猜测应删减或补齐哪些帧。`finalize` 省略已记录错误的失败集，发布其余成功结果。
