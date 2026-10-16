# ABC-130K 源格式说明

依据本地随数据下载的 [YAM_DATA_FORMAT.md](../../../dataset/raw/ABC-130K/docs/YAM_DATA_FORMAT.md)，仅描述原始数据。

## 文件组织

- 机器人为 YAM 双臂，每臂 6 个关节及一个平行夹爪。
- 每条 episode 一个目录，`episode.mcap` 保存状态、动作、视频、标定和任务指令；消息使用 Protobuf 编码。
- 有子任务标注时另有 `annotation.mcap`，没有标注时该文件不存在。
- `episode.mcap` 的 `session-metadata` 记录 episode ID、操作员 ID、任务指令及起止时间。

## 主要字段

以下 `{side}` 为 `left` 或 `right`。

| Topic | 主要内容 |
|---|---|
| `/{side}-arm-state` | `RobotState`：实测关节角 `position[6]`，顺序为基座到腕部的 `joint1…joint6`，单位 rad；末端位姿 `pose[16]` |
| `/{side}-arm-action` | `RobotState`：指令关节角和末端位姿；不是实测 state |
| `/{side}-ee-state` | `GripperState.position[0]`：实测夹爪开合，归一化到 `[0,1]`，`0` 闭合、`1` 张开 |
| `/{side}-ee-action` | `GripperState.position[0]`：指令夹爪开合 |
| `/instruction` | 单条 `Instructions` 消息，`data` 为整条 episode 的任务文本 |
| `/subtask-annotation` | 位于 `annotation.mcap`，`data` 为子任务文本；从本条时间戳生效到下一条，最后一条延续到 episode 结束 |

实测 arm 流还含 `velocity`、`torque`，官方说明为 6 个关节加夹爪共 7 维，读取时应检查实际长度；action 流这两项为空。夹爪实测流也有单独的速度和力矩字段。

## 位姿与坐标系

- `pose` 是按行展开的 4×4 齐次变换矩阵，表示末端在固定 world 系中的位姿；平移单位为米。
- world 系固定在工作站左臂基座侧，采用右手系：X 向前、Y 向左、Z 向上；左右臂共用此系。
- 官方说明未明确原始 `pose` 对应哪个末端 frame。公开[双臂模型](https://github.com/amazon-far/abc/blob/6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0/abc_sim/models/yam_bimanual_empty.xml#L182)中，左右臂均有 `tcp_site`（相对各自 `link_6`，位置 `[0,0,0]`）与 `grasp_site`（位置 `[0,0,0.1347]` 米）；两者旋转均为 `quat="1 0 0 -1"`，对应 site 的 X = −Y_link6、Y = +X_link6、Z = +Z_link6。这是模型定义，尚未证实原始 `pose` 使用其中哪一个，不能直接当作目标格式的 TCP/轴向定义。

## 时间戳

- MCAP `log_time` 为 Unix 绝对时间，单位纳秒；消息内 `timestamp` 为 `seconds + nanos`，两者一致。
- `session-metadata` 的 `start-time-unix`、`end-time-unix` 单位是**毫秒**。
- 两个 MCAP 文件共用绝对时间基准。各臂、夹爪和相机独立采样，频率和样本数可能不同，不能按行号配对；关联数据时按时间戳匹配。

## 相机

- RealSense 工作站：`/top-camera`、`/left-wrist-camera`、`/right-wrist-camera`，共 3 路，H.264，640×480。
- ZED-X 工作站：`/top-left-camera`、`/top-right-camera` 及两路腕部相机，共 4 路；顶部 H.265、腕部 H.264，1920×1200。
- 每条 `foxglove.CompressedVideo` 消息包含一帧的 Annex B 编码数据；根据各流 `format` 读取编码类型，帧率由实际时间戳确定，不写死。
- 有内参时存在对应的 `…-info` topic（`foxglove.CameraCalibration`），记录分辨率、内参 `K` 和畸变模型；没有内参的相机可缺少该 topic。
