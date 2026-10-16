# 10Kh-RealOmin-OpenData 源数据说明

上游：[Hugging Face 数据集卡](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData)、
[官方读取工具 das-datakit](https://github.com/genrobot-ai/das-datakit)。
转换入口为 `scripts/convert/realomin.py`；运行方式见[转换说明](conversion.md)。
下文区分原始字段、默认模型几何映射及尚未取得的实机标定。

## 文件与本地子集

一条轨迹对应一个 MCAP 文件，消息使用 Protobuf，schema 嵌入 MCAP。
原始目录按场景、技能及采集分组组织，夹爪资产为根目录的 `DAS_Gripper_V3.zip`。

本次固定 revision `fcbc0d38550e134f273426aa7c9cc2b491270bc4`，
在 `Cooking_and_Kitchen_Clean/clean_bowl/00001` 的 724 条轨迹中按 seed 42 抽取：

| 文件 | MCAP 消息覆盖时长 |
|---|---|
| `00032.mcap` | 103.47 秒 |
| `00246.mcap` | 53.93 秒 |
| `00307.mcap` | 91.35 秒 |
| `00526.mcap` | 32.25 秒 |
| `00557.mcap` | 52.25 秒 |

本地目录：`dataset/raw/10Kh-RealOmin-OpenData/`。
`subset_manifest.json` 记录固定版本、源路径与字节数；`inspection.json` 保存本地检查结果。
样本及夹爪资产共约 304 MiB。

## 原始字段

上游约定 `robot0` 为左夹爪，`robot1` 为右夹爪。两者各有以下消息：

| Topic 后缀 | 字段与内容 |
|---|---|
| `/sensor/camera0/compressed` | `data` 为 H.264 字节流；`header.timestamp` 为时间戳 |
| `/sensor/camera0/camera_info` | `width/height`、`K/D/R/P`、`T_b_c` 标定参数 |
| `/sensor/magnetic_encoder` | `value` 为夹指距离，官方声明范围为 0–0.103 米，0 为闭合 |
| `/vio/eef_pose` | `pose.position` 与 `pose.orientation.{x,y,z,w}`；`frame_id` 为 `world` |
| `/sensor/imu` | 角速度与线加速度 |

样本还包含 `/sim/robot_info` 和 `/system_info`。双路 H.264 实际解码尺寸为 1600×1300，
但样本 `camera_info.width/height` 为 640×480；转换 RGB 时须以解码帧尺寸为准，
标定参数与实际图像分辨率的对应关系需另行核对。位姿、开度与图像消息数量不相同。

## 时间与坐标

实际样本的 `header.timestamp` 是约 `1.765e18` 的整数纳秒。
转换读取 `header.timestamp`，不以 MCAP `log_time/publish_time` 替代，不转为浮点秒。
四路 state 与全部 RGB 的最早首样本共同归零；各路保留原生样本数量和间隔，不同步或插值。
右手消息中的 `header.topic_name` 有时仍写为 `/robot0/vio/eef_pose`，须以 MCAP channel 区分左右。

`/vio/eef_pose` 位姿与同刻 `/sim/robot_info.base_link.transform` 一致；
仅凭 topic 中的 `eef` 不能把它直接当成目标契约要求的 TCP。
当前样本未发现独立的语言标注文件，技能名称来自源路径 `clean_bowl`。

[官方 Matrix Studio 坐标说明](https://docs.genrobot.ai/products/matrix-studio#_5-data-instruction)
将 `eef_pose` 定义为设备在固定 VIO 世界系下的位姿，局部 X 向前、Y 向左、Z 向上；
世界原点取初始相机中心，Z 反重力。转换保留各手原生世界系，不要求重建左右手之间的外参。

## 默认工具变换与开度

默认固定变换为 `T_world_tcp = T_world_source · T_source_tcp`。
以下载的 V3 URDF 为几何参考，`joint_1=0.925` 的闭合位形中，
`link_3/link_4` 网格指尖内缘中心的中点作为固定 TCP 估计：

```text
p_source_tcp = [0.14405256, 0.00049106, -0.03820249] m
R_source_tcp = Ry(-15°)
```

规范接近方向沿手指固定伸出方向，掌背法向为该旋转的 Z 列；
TCP 固定在上述闭合位形定义的位置，不能随每帧开度改变。
此估计来自本次固定 revision 的 `DAS_Gripper_V3.zip` 中 URDF/网格，
不是设备厂商提供的 TCP 标定。当前将 VIO 局部 body 与该 URDF 的 base_link 轴向及原点视为一致，
官方说明中的相机光心与 URDF 装配原点仍缺少经测量的注册外参。
`camera_info` 分辨率及外参也与这份装配模型不完全一致，不能用它消除此项不确定性。
精确物理 TCP 和平移 delta 需取得实机外参后用 `--tcp-offset`、`--tool-rpy` 重导。

实际夹爪值按 `openness = clip(value / 0.103, 0, 1)` 转换；
0 米闭合、0.103 米张开，采用数据集卡声明的范围，不按样本极值重标定。
非有限值报错；有限的微小越界（样本可略高于 0.103）裁剪。

指令默认取最近的非数字父目录技能名，将下划线换为空格；这属于源技能标签，
不是额外生成的逐帧语言标注。可用 `--instruction` 指定已有的任务文本。
