# ABC-130K

| | |
|---|---|
| 上游 | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) |
| 数据集键 | `abc130k` |
| 位姿来源 | 实测关节角经 YAM 模型 FK 计算 |
| 机器人模型 | `assets/robot_models/yam` |
| 不合格的 episode | 报错，记为失败 |

```bash
loom-data download abc130k select --per-task 30
loom-data download abc130k download
loom-data convert run --dataset abc130k --workers 8
```

通用选项、并行与续跑见[使用说明](../usage.md)；输出格式见[导出契约](../contract/export-contract.md)。

## 原始数据

原始格式依据随数据下载的 `docs/YAM_DATA_FORMAT.md`。

### 文件组织

- 机器人为 YAM 双臂，每臂 6 个关节及一个平行夹爪。
- 每条 episode 一个目录，`episode.mcap` 保存状态、动作、视频、标定和任务指令；消息使用 Protobuf 编码。
- 有子任务标注时另有 `annotation.mcap`，没有标注时该文件不存在。
- `episode.mcap` 的 `session-metadata` 记录 episode ID、操作员 ID、任务指令及起止时间。

### 主要字段

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

### 位姿与坐标系

- `pose` 是按行展开的 4×4 齐次变换矩阵，表示末端在固定 world 系中的位姿；平移单位为米。
- world 系固定在工作站左臂基座侧，采用右手系：X 向前、Y 向左、Z 向上；左右臂共用此系。
- 官方说明未明确原始 `pose` 对应哪个末端 frame。公开[双臂模型](https://github.com/amazon-far/abc/blob/6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0/abc_sim/models/yam_bimanual_empty.xml#L182)中，左右臂均有 `tcp_site`（相对各自 `link_6`，位置 `[0,0,0]`）与 `grasp_site`（位置 `[0,0,0.1347]` 米）；两者旋转均为 `quat="1 0 0 -1"`，对应 site 的 X = −Y_link6、Y = +X_link6、Z = +Z_link6。这是模型定义，尚未证实原始 `pose` 使用其中哪一个，不能直接当作目标格式的安装接口原点/轴向定义。

### 时间戳

- MCAP `log_time` 为 Unix 绝对时间，单位纳秒；消息内 `timestamp` 为 `seconds + nanos`，两者一致。
- `session-metadata` 的 `start-time-unix`、`end-time-unix` 单位是**毫秒**。
- 两个 MCAP 文件共用绝对时间基准。各臂、夹爪和相机独立采样，频率和样本数可能不同，不能按行号配对；关联数据时按时间戳匹配。

### 相机

- RealSense 工作站：`/top-camera`、`/left-wrist-camera`、`/right-wrist-camera`，共 3 路，H.264，640×480。
- ZED-X 工作站：`/top-left-camera`、`/top-right-camera` 及两路腕部相机，共 4 路；顶部 H.265、腕部 H.264，1920×1200。
- 每条 `foxglove.CompressedVideo` 消息包含一帧的 Annex B 编码数据；根据各流 `format` 读取编码类型，帧率由实际时间戳确定，不写死。
- 有内参时存在对应的 `…-info` topic（`foxglove.CameraCalibration`），记录分辨率、内参 `K` 和畸变模型；没有内参的相机可缺少该 topic。

## 转换规则

视频解码与编码全部在 CPU 上执行，需要 PyAV 18.1 以上，不需要 NVIDIA GPU 或
额外 ffmpeg 命令行程序。逐帧解码源 H.264/H.265，缩放到契约分辨率后用 libx264
重新编码为无音轨 H.264 MP4，使用 preset fast、CRF 18，恒定 30 fps。MCAP 读取与 MP4
封装同样在 CPU 上执行。

### 关节 FK 与工具坐标系

仅使用 `/{left,right}-arm-state.position` 六维实测关节角，依次映射到模型
`arms[side].joints`，以弧度输入 URDF FK；不读取原始 `pose`，不使用 action。
参考系为模型的固定 `base_link`；左右臂在自己的原生时间戳分别计算，无需同步另一臂。

EEF 原点使用夹爪安装接口，即 `arm_{side}_link6` 的原点。实现选取与其位置重合的
模型 `arm_{side}_tcp` site；不再使用向指尖偏移 `0.1347 m` 的 `grasp` site。
更换指尖不改变这个原点。

EEF 三轴按契约的固定规则对齐到 YAM 模型：

- `+X = +Z_link6`：工具接近方向，从 `link6` 原点指向 TCP。
- `+Z = +Y_link6`：掌背法向。模型展示位姿下 `+Y_link6` 即 `base_link` 的 `+Z`（正上方），
  腕部 D405 相机也位于该侧。
- `+Y = +Z × +X = +X_link6`：夹爪开合轴，指向 finger2 一侧。

模型 tcp site 自带旋转为 X = −Y_link6、Y = +X_link6、Z = +Z_link6，
因此从 tcp site 到规范 EEF 系右乘：

```text
[ 0  0 -1 ]
[ 0  1  0 ]
[ 1  0  0 ]
```

结果写为位置三维加完整旋转矩阵三列（列优先展平）。以上采用模型几何与关节定义，
并不表示已完成实物零位标定；不需要提供从原始 pose 到安装接口的变换配置。

### 同步、视频与指令

- 腕部相机分别导出为 `left_wrist`、`right_wrist`。RealSense 的 `/top-camera` 和 ZED-X 的左目 `/top-left-camera` 各自在对应工作站导出为 `top`。
- 源是异步采样，按契约第 3 节重采样到 30 fps：帧序列覆盖四路 state 和三路相机都有数据的时间段；
  位姿做位置线性插值加旋转 slerp，夹爪线性插值，RGB 取采集时刻最近的原生帧。
  帧时刻两侧的 state 样本间隔超过 100 ms 或最近 RGB 帧偏离超过 50 ms 的 episode 不导出，
  记为 `sampling_gap`；各流的共同时间段不足两帧的记为 `no_common_interval`。
- FK 在每个原生关节样本上计算，再对得到的位姿插值，不对关节角插值。
- 夹爪实测开合裁剪到 `[0,1]`：小于 0 置为 0，大于 1 置为 1，范围内保持原值。
  NaN、正负无穷等非有限值仍在视频转码前报错。本地样本中存在少量超过 1 的夹爪读数
  （例如 `episode_e18a509f-5940-49ca-815b-a27ebfdcd4b1` 右夹爪最大
  `1.0023205498338077`），导出时按上述规则裁剪，不修改原始数据。
- 按各相机消息的 `format` 解码 H.264/H.265，逐帧编码成无音轨 H.264 MP4。
  用解码 PTS 将显示帧映射回输入消息；校验每个输入样本恰好对应一帧，再按上面的规则选帧。
- 无子任务时，所有帧使用整集任务。
  有 `annotation.mcap` 时，子任务从标注时刻开始生效并替代整集指令，
  第一条子任务前保留整集任务；同一时刻最后一条标注生效。每帧取其时刻所在区间的指令。
- 校验 Protobuf 时间戳与 MCAP `log_time` 一致。缺失 state、相机或任务文本时报错。

### 布局

允许的布局只有一种相机集合：`top`、`left_wrist`、`right_wrist` 三路齐全。缺任何一路的 episode 不导出，记为 `unsupported_layout`。

两种工作站的相机宽高比不同，自动导出为两个数据集：

| 工作站 | 源分辨率 | 布局 |
|---|---|---|
| RealSense | 640×480（4:3） | `top+left_wrist+right_wrist_640x480` |
| ZED-X | 1920×1200（16:10） | `top+left_wrist+right_wrist_640x400` |
