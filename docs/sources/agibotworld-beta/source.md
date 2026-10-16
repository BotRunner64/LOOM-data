# AgiBot World Beta 源数据说明

上游：[Hugging Face 数据集卡](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)。
Beta 使用 HDF5、MP4 与任务 JSON，与本项目已有的 AgiBotWorld2026 输入格式不同。
转换入口为 `scripts/convert/agibotworld_beta.py`，运行方式见[转换说明](conversion.md)。

## 仓库和子集结构

上游状态文件和相机参数按全局 episode 范围打包为 TAR，视频按任务打包为 TAR：

```text
proprio_stats/<start>-<end>.tar       # 内部为 <task>/<episode>/proprio_stats.h5
observations/<task>/<start>-<end>.tar # 内部为 <episode>/videos/*.mp4 和 depth/
parameters/<start>-<end>.tar
task_info/task_<task>.json
```

起止编号只是分片范围，内部 episode 是稀疏的，跨分片也存在范围重叠。
下载脚本实际解析 TAR 成员，以任务及 episode 两个编号同时匹配。

本次固定 revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`，
从 `proprio_stats/866961-940165.tar` 与 `observations/694/881816-883115.tar`
选取 task 694（`Twist the bottle cap`）的 5 条轨迹：

| Episode | 状态行数 / 每路 RGB 帧数 |
|---|---|
| `881865` | 2000 |
| `881883` | 1825 |
| `881898` | 1848 |
| `881984` | 1892 |
| `883115` | 1878 |

本地输出在 `dataset/raw/AgiBotWorld-Beta/`：

```text
proprio_stats/694/<episode>/proprio_stats.h5
observations/694/<episode>/videos/*.mp4
task_info/task_694.json
subset_manifest.json
inspection.json
.archives/                          # 用于索引及重复下载的源分片
```

选中的 HDF5/RGB 约 254 MiB。两个有效源分片约 3.55 GiB，
另有选择时检查的约 253 MiB 分片未匹配到状态，仍作为缓存保存在 `.archives/`。
当前未提取 depth，也未下载 parameters。

## HDF5 字段

依据上游数据集卡，状态左右顺序均为左在前、右在后：

| 字段 | 实际样本 shape | 上游声明的语义 |
|---|---|---|
| `timestamp` | `[N]`, int64 | 纳秒时间戳 |
| `state/end/position` | `[N,2,3]` | 双臂法兰位置，米 |
| `state/end/orientation` | `[N,2,4]` | 法兰四元数，xyzw |
| `state/effector/position` | `[N,2]` | 数据页称开度毫米；本次实际读数随闭合增大，见下文 |
| `state/robot/position` | `[N,3]` | odom 系底盘位置，米 |
| `state/robot/orientation` | `[N,4]` | 底盘 xyzw 四元数 |
| `state/waist/position` | `[N,2]` | pitch（弧度）与 lift（米） |

HDF5 还包含关节、head、wrench 和 action 字段，部分数组为空。
本次是双夹爪样本；上游另有 `state/effector/position` 为 `[N,12]` 的灵巧手样本。

## RGB 与任务标注

每条样本均有 8 路 AV1 MP4（PyAV 解码器名为 `libdav1d`），完整解码后各路帧数等于同集 `timestamp` 行数：

- `head_color`、`hand_left_color`、`hand_right_color`：640×480。
- `head_left_fisheye_color`、`head_center_fisheye_color`、`head_right_fisheye_color`、
  `back_left_fisheye_color`、`back_right_fisheye_color`：960×768。

MP4 容器帧率为 30，PTS 从 0 开始。容器时间不是原始采集时间；
转换使用 HDF5 的整数 `timestamp` 作为对应显示帧的采集时间。
该映射与[上游同格式 Alpha 转换器](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/blob/main/scripts/convert_to_lerobot.py)
的 `load_local_dataset` / `save_episode(videos=...)` 一致：同一 episode 的逐行状态和各路视频按帧序对应。
Beta 样本的 40 路视频全部解码后也与本集行数一致。

`task_694.json` 是 episode 对象列表，实际键为 `label_info`，
其中 `action_config` 提供 `start_frame/end_frame/action_text/skill`。
各段共用帧边界，例如前段 `end_frame ==` 后段 `start_frame`。
转换取 `[start_frame,end_frame)`，映射到 HDF5 原始时间戳；最后一段的 `end_frame=N-1`
正好对应目标契约的整集终点（最后采样时刻），也兼容上游将整集终点写为 N 的形式。
空白处使用 `task_name`，空的 `action_text` 不替代任务文本；重叠或越界的有效标注报错。

## 固定参考系与 TCP

本次真实样本的法兰位姿已经包含腰部运动，参考系为源底盘 base，而非 `arm_base_link`。
独立核对采用[官方 G1_120s URDF](https://github.com/AgibotTech/genie_sim/blob/6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46/source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg/G1_120s/G1_120s.urdf)
对同刻腰部与双臂关节做 FK：第 881865 集的 6 个真实采样中，旋转相符，
平移仅有时不变的 Z=-0.3 米参考原点差（其余残差约微米量级）。
样本与独立 FK 数值保存在 `tests/fixtures/agibotworld_beta_pose_samples.json`。
不能再左乘一次腰部 FK，也不能套用 2026 的五关节腰部链。

```text
T_fixed_tcp = T_odom_base(source) · T_base_flange(source) · T_flange_tcp
```

有效 odom 位置和 xyzw 姿态照常使用。底盘位置与四元数同时全零时，参照 2026 的缺失标记规则，
逐集核对 `action/robot/velocity` 的两维完整、有限且绝对值不超过 `1e-6`：整集缺失使用
固定底盘参考系，部分缺失沿用最近实测位姿（开头缺失用首个实测位姿）。
位置非零而四元数全零、其它无效姿态，以及缺失位姿但速度命令不满足条件均整集 drop。
速度命令只用于此项核验，不导出为 state，也不积分重建运动底盘。
源参考原点的时不变 Z 偏移保留，不影响 body-frame delta 的定义。

服务器按 233 个任务各抽一集，50 集底盘位置与四元数整集全零，速度命令也整集为零。
183 集有效底盘样本中，未发现速度命令全零却存在明显底盘运动（平移超过 0.1 米或转角超过 10 度）的反例。
这是抽样核验，不代表全量比例；恢复条件在转换时逐集检查。

默认 TCP 平移为 `[0,0,0.207056] m`，采用公开 G1_120s 长指模型闭合指垫中心的几何估计，
不是本次实机夹爪的官方 TCP 标定。接近方向在源法兰中为 +Z；
模型左夹爪的掌背取 +X，右夹爪因安装关节绕 Z 转 π 而取 -X。
规范 X/Y/Z 列依次为：

```text
left:  [[0, 0, 1], [0,-1,0], [1,0,0]]
right: [[0, 0,-1], [0, 1,0], [1,0,0]]
```

固定工具安装及 TCP 对真实批次的注册尚未取得官方外参。
精确物理 TCP 使用 `--tcp-offset` 和左右 `--*-tool-rpy` 提供实机标定后重导；
独立 FK 回归只验证法兰参考系，不验证这个工具几何估计。

## 实际夹爪方向和默认基准

第 881865 集第 0 帧左右实测约 35.73/35.11、控制值均为 0，腕部 RGB 显示张开；
第 250 帧左手实测约 117.07、控制值为 1，RGB 显示抓住瓶身。
实际读数随闭合增大，与数据卡的“开口毫米”描述不一致，不能直接 `value/120`。
转换只输出实测 state，action 仅用于这次语义核对。

默认 `opened=35`、`closed=120` 是依据该批控制状态稳定平台及腕部视频选择的**暂定源读数基准**，
不是已标定的夹指距离，也不是每集的最小最大值。归一化为：

```text
openness = clip((value - 120) / (35 - 120), 0, 1)
```

提供 `--gripper-open`、`--gripper-closed` 覆盖该基准，参数单位与源读数一致；
该线性标量表示执行器开合程度，不能当成精确指尖距离。跨夹爪型号/固件必须核对基准。
灵巧手 `[N,12]` 不能满足目标的两路标量开度，转换整集 drop 并单独统计。
