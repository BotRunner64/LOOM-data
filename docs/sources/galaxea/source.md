# Galaxea Open-World Dataset 源数据契约

本文描述本项目已下载的 LeRobot v2.1 数据；版本由 `dataset/raw/Galaxea-Open-World-Dataset/subset_manifest.json` 指定。转换方法见[转换说明](conversion.md)，输出只遵循[目标契约](../../contract/export-contract.md)。

## 目录与本体

每个任务归档解压后是独立数据集：

```text
<task>/
├── meta/info.json
├── meta/episodes.jsonl
├── meta/tasks.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.<camera>/episode_000000.mp4
```

`info.json` 的 `data_path`、`video_path` 和 `chunks_size` 定义文件寻址规则。`episodes.jsonl` 提供 `episode_index`、`length`；`tasks.jsonl` 提供索引到文本的映射。不同任务的 episode 索引会重复。

以每个任务包的 `robot_type` 为准，不能将整个数据集默认视为 R1 Lite。当前本地包括：

| robot_type | 本体 | 单臂关节数 | 本地任务包 / episode |
|---|---|---:|---:|
| `r1lite` | R1 Lite / A1X | 6 | 4 / 224 |
| `r1pro` | R1 Pro / A2 | 7 | 1 / 20 |

官方数据页仍有统一 R1 Lite 的描述，与本地收纳盒任务包的 `r1pro` 元数据和 7 维关节记录不同。

## 采样字段

| 字段 | 内容 |
|---|---|
| `observation.state.{left,right}_arm` | 实测关节角，分别为 6 或 7 维；用于本体核验，不作为导出 state |
| `observation.state.{left,right}_ee_pose` | 7 维 `[x,y,z,qx,qy,qz,qw]`，位置为米 |
| `observation.state.torso` | 4 维；R1 Lite 前三维为关节角，第四维为零占位；R1 Pro 四维均为关节角 |
| `observation.state.{left,right}_gripper` | G1 夹爪实测开口行程，0 闭、100 开，毫米；元数据声明 `[1]`，本地 Parquet 实际为标量 |
| `timestamp` | episode 内相对秒数，源为 float32 |
| `frame_index` | 本集从 0 开始的连续帧序号 |
| `task_index` / `coarse_task_index` | 本行细粒度 / 整体任务文本索引 |
| `quality_index` / `coarse_quality_index` | 质量标签索引，不是任务指令 |

源 LeRobot 已将状态与图像对齐到逐帧表。本地只有这一列共同时间戳，没有原始 ROS 各 topic 的独立采样时间。转换保留此发布格式的时间戳及全部样本，不能由此恢复原始异步采集时间；不按帧号、15 FPS 或视频 PTS 重造时间。

相机为 `head_rgb`、`head_right_rgb`、`left_wrist_rgb`、`right_wrist_rgb`，每集各自一个 MP4；本地为 AV1、15 FPS。每路显示顺序的帧数应与逐帧表相等。头部双目左右图像分别保存。

## 原生末端参考系

下面的映射来自本地实测关节与对应官方模型 FK 的交叉核验，属于数据解释，不是实机标定：

| 本体 | ee_pose 的参考系 | ee_pose 的工具帧 |
|---|---|---|
| R1 Lite | 模型 `torso_link3` | 对应侧 `arm_*_link6`（法兰） |
| R1 Pro | 模型 `torso_link4` | 对应侧 `arm_*_gripper_base_link` |

两者参考系均随躯干运动，不是固定世界系。转换使用实测躯干关节做 FK，将原生末端位姿变换到底盘系；不使用模型重算手臂位姿。实测关节与末端 topic 存在小的同步差异，不能要求每行 FK 严格相等。

`tests/fixtures/galaxea_pose_samples.json` 保留 5 个任务第 0 集的首帧、约 1/3、约 2/3 和末帧共 20 个真实样本，回归检查上述工具帧区别、躯干组合与旋转表达。该检查不验证固定 TCP 本身的实机精度。

## G1 TCP 与工具轴

统一 TCP 暂定义为两指工作端尖端的中点，开合时保持不变。由当前模型几何：夹爪前端基座到指 link 原点为 `0.03689 m`，指网格向工作端的延伸为 `0.041465 m`，合计 `0.078355 m`。R1 Lite 原生法兰到夹爪前端基座还需增加 `0.08165 m`。

| 本体 | TCP 平移（在原生 ee_pose 中，米） | 规范 +X 接近方向 | 规范 +Z 掌背方向 | 规范 +Y |
|---|---|---|---|---|
| R1 Lite | `[0.160005, 0, 0]` | 原生 +X | 原生 +Z（腕部相机安装侧） | 原生 +Y |
| R1 Pro | `[0, 0, -0.078355]` | 原生 −Z | 原生 +X（腕部相机安装侧） | 原生 +Y |

R1 Lite 轴变换为单位矩阵；R1 Pro 的 `R_native_contract` 各列为规范 X、Y、Z 在原生工具系中的向量：

```text
[[ 0, 0, 1],
 [ 0, 1, 0],
 [-1, 0, 0]]
```

这些 TCP 是显式的模型几何估计，未取得该批实机工具标定。两种模型保持同一物理定义；不能把 R1 Lite 法兰偏移直接套到 R1 Pro 原生夹爪系。型号、指尖附件或标定变化时应更新对应固定工具变换后重导。

## 底盘记录的边界

`observation.state.chassis` 来自 `/hdas/feedback_chassis.position[0:3]`，表示三轮转向角；`observation.state.chassis.velocities` 是三轮实测线速度。它们不是底盘位置或机体 `[vx,vy,wz]`。`action.chassis.velocities` 为目标速度命令，不是实测速度。本地旧样本为六维 twist `[vx,vy,vz,wx,wy,wz]`；2026-09-27 检查服务器挂载的 227 个任务，均为三维 `[vx,vy,wz]`，字段 names 对应 linear.x、linear.y、angular.z。转换根据 meta/info.json 的 shape 检查行维度，并要求所有分量为零。服务器版本的 `observation.state.chassis` 还将三轮 position 和 velocity 合并为六维；该反馈字段不参与筛选。

`observation.state.chassis.imu` 包含姿态四元数、角速度和加速度，但不能代替完整底盘平移记录。本地样本发现姿态跳变和待核实的单位，因此本转换不使用它推算底盘运动。

本地 LeRobot 未提供可直接使用的底盘 odom/map 位姿。转换只接收全程底盘速度命令为零（容差 `1e-6`）的轨迹，假定底盘固定并以其作为参考系；轮速反馈不参与筛选；轮速积分、IMU 融合和移动底盘轨迹不在此转换范围内。

## 指令

`tasks.jsonl` 同时包含动作文本、`null` 占位及质量标签。不可使用 `episodes.jsonl.tasks[0]` 作为整集指令，因为它可能是 `qualified` / `unqualified`。

逐行优先使用 `task_index` 对应的文本；为空或 `null` / `none` / `nan` 时使用该行 `coarse_task_index`。保留中英双语文本及 `@` 分隔符。质量标签不能作为指令。连续相同文本合并，变化边界使用源行时间；最终区间截止到本集最后样本时间。

## 来源

- [官方数据页](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
- [R1 Lite 官方接口：躯干、轮子反馈和夹爪行程](https://docs.galaxea-ai.com/Guide/R1Lite/software_introduction/ros2/R1Lite_Software_Introduction_ros2/)
- [R1 Pro 官方接口：七关节手臂与四关节躯干](https://docs.galaxea-ai.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/)
- [官方 GalaxeaManipSim 模型](https://github.com/OpenGalaxea/GalaxeaManipSim/tree/abe7f5161eeaa150e6eaffdf443af5df7f23f356/galaxea_sim/assets)，本地资产来源与哈希见各模型 `robot.json`。
