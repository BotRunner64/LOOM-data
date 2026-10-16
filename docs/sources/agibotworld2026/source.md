# AgiBot World 2026 源数据契约

本项目下载的 `ImitationLearning/CommercialSpaces/task_*/<起止帧>.tar.gz` 是各自独立的 LeRobot v2.1 数据集。以下记录的是已下载版本的实际结构与转换所用字段；源数据版本见 `dataset/raw/AgiBotWorld2026/subset_manifest.json`。

- `data/meta/info.json`：`robot_type=g2a`，视频相机特征、`observation.state` 向量的 `field_descriptions[*].indices`，以及可选的 `instruction_segments`。各归档的状态向量长度不同，字段索引必须从本归档的 `info.json` 读取。
- `data/meta/episodes.jsonl`：每集任务文本和长度。
- `data/data/chunk-*/episode_*.parquet`：逐帧 `observation.state`、`timestamp`（相对本集起点的秒数，源列为 float32）、`frame_index`、`episode_index`。状态与相机按帧对应；没有每个相机独立的采集时间列。源秒数乘 `1e9` 后取整，不按 30 fps 重造时间戳；由于源为 float32，后段时间存在微秒级量化误差。
- `data/videos/chunk-*/observation.images.*/episode_*.mp4`：逐相机视频。深度通道由 `video.is_depth_map=true` 标注；RGB 通道为 AV1 MP4。每个视频按显示顺序对应同集 Parquet 的每一行。

转换读取实测夹爪 `state/left_effector/position`、`state/right_effector/position`，双臂末端 `state/end/arm_position`（左 xyz、右 xyz，米）和 `state/end/arm_orientation`（左 xyzw、右 xyzw），`state/waist/position` 的 5 个腰部关节，以及存在时的 `state/robot/position` 和 `state/robot/orientation`（xyzw）。不把 `action` 写入目标格式。

### 末端参考系与工具变换

`state/end/arm_*` 按相对 `arm_base_link` 的法兰位姿解释。依据是[官方 G2 IK/FK 工具的坐标约定](https://github.com/AgibotTech/genie_sim/blob/6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46/source/geniesim_benchmark/src/geniesim_benchmark/utils/g2_ikfk_converter.py)以及本地数据交叉核验：源末端左乘腰部 FK 后，源腕部相机相对法兰的位置在第 1 集每 10 帧抽样中，各分量标准差小于 `4e-8 m`。这验证源字段与腰部链的相容性，并不证明相机或 TCP 已对真实图像精确标定。

转换链（`T_A_B` 将 B 系映到 A 系）：

```text
T_fixed_tcp = T_fixed_base · T_base_arm_base(waist) · T_arm_base_flange(source) · T_flange_tcp
```

本地 `crsB + omnipicker` 模型仅用于 `base_link → arm_base_link` 腰部链。其手臂长度与源末端记录不相容，不能用它重算双臂末端；`arm_base_link` 随腰部运动，也不能直接作为导出的固定参考系。

规范轴在源法兰系中取 `X_tcp=+Z_flange`、`Z_tcp=+X_flange`（腕部相机安装侧，即掌背侧）、`Y_tcp=-Y_flange`。因此 `R_flange_tcp` 的列为：

```text
[[0,  0, 1],
 [0, -1, 0],
 [1,  0, 0]]
```

**TCP 尚未取得该批 G2 实机的官方标定。** 默认平移暂取 `[0, 0, 0.207056] m`，来自[官方公开 G1_120s 模型的 CRT120S 长指夹爪几何](https://github.com/AgibotTech/genie_sim/blob/6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46/source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg/G1_120s/G1_120s.urdf)：零位两侧指垫 link 原点的中点，轴向长度为 `0.10644 + 0.0129 + 0.04341 + 0.044306`。视频中的长指外形及相机投影支持用它作预览估计，但源 `g2a` 元数据没有确认夹爪型号、安装变换和实机 TCP，不能把这个值视为精确物理标定。固定 TCP 不随开合改变；得到实机标定后应使用 `--tcp-offset X Y Z` 重导。当前数据可用于检查运动，精确 TCP 位置及由它计算的平移 delta 仍有此项不确定性。

本地 `task_3401/399093_399454.tar.gz` 将底盘位置与姿态声明为 `dimensions: 0, indices: []`，即整个归档的 14 集都没有底盘位姿记录。该归档共 38,345 帧的 `action/robot/velocity` 两维全部为 0；结合这一归档为原地操作的确认，转换时将底盘视为不移动，使用固定底盘参考系。脚本逐集校验速度命令确实为零；其它有底盘位姿的归档照常使用实测位姿。若未来遇到缺少位姿且底盘命令非零的归档，转换会报错。

本地 5 个归档的实测夹爪值在约 `[-0.91, 0]`。对照 `task_3401` 第 1 集腕部 RGB：第 0 帧两手源值为 `0`，夹爪张开；第 1000 帧源值约 `[-0.64064, -0.63882]`，夹爪抓持物体。因此采用 `openness = clip(1 + state/0.91, 0, 1)`；`0` 源值对应张开，`-0.91` 对应归一化闭合端。`0.91` 是本地样本幅度界限；线性映射不是指尖距离标定。不能套用本地 OmniPicker 模型主动关节的开合符号。跨型号或数据版本需重新核对。

逐集指令优先选 `instruction_segments` 的 `default` track：其区间是 `[start_frame_index, end_frame_index)`；其它 track 可能交叠，不并入单一指令时间轴。`default` 的空白区间与没有 `default` track 的整集使用 `episodes.jsonl.tasks[0]`。末尾 `end_frame_index == episode_length` 映射到最后一帧的时间戳，以满足目标契约的整集终点定义。

完整模仿学习数据中还存在底盘字段有索引、但底盘位置和四元数一起全零的缺失记录。例如 `task_3777/368929_370517.tar.gz` 的 `episode_000001` 整集全零；`task_3402/375260_377477.tar.gz` 的 `episode_000056` 则前 1655 帧有有效位姿，后 858 帧全零。根据该批数据的使用者确认，未移动的底盘可能不记录位姿；转换器验证整集底盘速度命令全零后，整集缺失时使用固定底盘参考系，部分缺失时沿用之前最近的实测底盘位姿（开头缺失时使用首个实测位姿）。有效实测位姿保留；位置非零而四元数全零、其它无效四元数或缺失位姿时底盘速度命令非零的情况仍报错。

源格式说明：[AgiBot World 2026 官方数据页](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)。
