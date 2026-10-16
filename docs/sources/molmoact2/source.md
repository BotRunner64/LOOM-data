# MolmoAct2-BimanualYAM 源格式说明

依据[官方数据卡](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset)、
下载保留的 `source/meta/info.json`、任务表及逐帧 Parquet。这里只描述源数据语义；
转换规则见 [转换说明](conversion.md)。

## 文件组织

合并后的 LeRobot v3.0 数据集，`robot_type = bi_yam_follower`。
原仓库的 `data/chunk-*/file-*.parquet` 和各相机 MP4 可容纳多条 episode。
`meta/episodes/` 记录 episode 的数据索引范围及各视频内的片段边界。
转换器可直接读取这个完整本地布局，无需先用下载器拆分 episode。

本项目 `scripts/download/molmoact2_subset.py` 的下载产物为：

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

## 状态与夹爪

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

## 时间和相机

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

## 语言

优先通过 `meta/tasks_annotated.parquet` 的 `episode_index → task` 获取整集详细标注。
官方数据卡要求在缺少有效标注时回退到逐行 `task_index` 查
`meta/tasks.parquet`；任务文本存于该表的 Pandas 索引列 `__index_level_0__`。
下载器还会将任务和标注缓存进 `episode.json`。

## YAM 工具几何约定

固定安装的双臂可分别以自身 `arm_{side}_base_link` 为时不变参考系。
仓库 YAM 模型的双臂外参来自 ABC 仿真布局，不代表 MolmoAct2 的安装标定。

沿用同一 YAM 工具的模型定义：TCP 为 `link6` 的 `[0,0,0.1347]` 米，
即模型 `grasp` site；规范工具轴为 `+X = +Z_link6`（接近方向）、
`+Z = +Y_link6`（相机侧掌背法向）、`+Y = +X_link6`。
从原生 grasp site 到规范工具系的固定变换无平移，旋转为：

```text
[ 0  0 -1 ]
[ 0  1  0 ]
[ 1  0  0 ]
```

模型依据和 site 定义见 [ABC 源说明](../abc130k/source.md) 与
[YAM 工具说明](../abc130k/conversion.md#关节-fk-与工具坐标系)。
此处采用名义模型几何和关节定义，不代表已逐条标定真实机器人零位、工具长度或安装外参。
