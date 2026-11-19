# 10Kh-RealOmin-OpenData

| | |
|---|---|
| 上游 | [genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData)（[官方读取工具](https://github.com/genrobot-ai/das-datakit)） |
| 数据集键 | `realomin` |
| 位姿来源 | 源记录的 VIO 位姿，乘固定工具变换 |
| 机器人模型 | 不需要 |
| 不合格的 episode | 源数据异常整集 drop 并记录原因；程序和磁盘错误仍记为失败 |

```bash
loom-data download realomin select --count 5
loom-data download realomin download
loom-data convert run --dataset realomin --workers 8
```

本数据源的选项（均可省略）：

| 选项 | 含义 | 默认 |
|---|---|---|
| `--eef-offset X Y Z` | 安装接口在 raw EE 系中的坐标，米 | 模型推导的 V3 后端安装点 |
| `--tool-rpy R P Y` | 规范工具轴相对 raw EE 的旋转，弧度 | `0 -π/12 0` |
| `--gripper-open / --gripper-closed` | 全开 / 闭合时的源读数，米 | `0.103` / `0` |
| `--instruction TEXT` | 所有 episode 使用的任务文本 | 路径中的技能名 |

通用选项、并行与续跑见[使用说明](../usage.md)；输出格式见[导出契约](../contract/export-contract.md)。

## 原始数据

### 文件与本地子集

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

### 原始字段

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

### 时间与坐标

实际样本的 `header.timestamp` 是约 `1.765e18` 的整数纳秒。
转换读取 `header.timestamp`，不以 MCAP `log_time/publish_time` 替代，不转为浮点秒。
四路 state 与两路 RGB 各自独立采样，转换按契约第 3 节重采样到 30 fps，见下文「同步与视频」。
右手消息中的 `header.topic_name` 有时仍写为 `/robot0/vio/eef_pose`，须以 MCAP channel 区分左右。

`/vio/eef_pose` 位姿与同刻 `/sim/robot_info.base_link.transform` 一致；
仅凭 topic 中的 `eef` 不能把它直接当成目标契约要求的夹爪安装接口。
当前样本未发现独立的语言标注文件，技能名称来自源路径 `clean_bowl`。

[官方 Matrix Studio 坐标说明](https://docs.genrobot.ai/products/matrix-studio#_5-data-instruction)
将 `eef_pose` 定义为设备在固定 VIO 世界系下的位姿，局部原点为中央相机光心，X 向前、Y 向左、Z 向上；
世界原点取初始相机中心，Z 反重力。转换保留各手原生世界系，不要求重建左右手之间的外参。

## 转换规则

### 安装接口变换与开度

目标 EEF 原点采用 V3 夹爪后端的名义安装点，轴向沿用原有定义。
转换仍为 `T_world_eef = T_world_raw · T_raw_eef`，不依赖夹爪开度或指尖长度。

#### 模型依据与坐标注册

模型来自厂商公开的 [DAS-Gripper Version Changelog](https://zcnma1sv5kma.feishu.cn/wiki/CKpbwye45iOlrckukIPc5hKCndh)
中 V3 行的两个附件：

| 附件 | URDF | URDF SHA-256 |
|---|---|---|
| `DAS_Gripper_V3.zip` | `DAS_Gripper_V3/urdf/DAS_Gripper_V3.urdf` | `dab31fdcd0978b2c49e014981eee119988ae41eb484a21c07f496abc95fd160c` |
| `DAS_Controller_V3_with_flange.zip` | `DAS_Controller_V3_with_flange/urdf/DAS_Controller_V3_with_flange.urdf` | `45fcd5f7585a25933d6a1e471024ac7e7f9a7e8328e4502926f8944f74ca77db` |

采集夹爪没有独立的法兰 link；配套控制夹爪有 `joint_Flange → link_Flange`，
固定平移为 `[-0.067889, -0.000038794, -0.097619] m`。
不能直接把这组数套到采集夹爪，因为两套模型的 `base_link` 基准略有不同。

本项目把该法兰点转移为采集夹爪的**等效后端安装基准**，采用以下名义注册：

1. 采集模型的 `joint_1/joint_2` 对应控制模型的 `joint1/joint2`，均为双指驱动轴。
2. 以两个驱动轴原点的中点对齐平移，以第一驱动轴的固定 RPY 对齐旋转。
   对应轴原点注册后的最大残差为 `0.0001 m`，说明这是名义几何对应而非完全相同的 CAD。
3. 将控制模型的 `joint_Flange` 原点变换到采集模型 `base_link`，得到：
   `[-0.069069831741, 0.000491298867, -0.099666510099] m`。
4. 按[官方 raw EE 定义](https://docs.genrobot.ai/products/matrix-studio#_5-data-instruction)，
   局部原点是中央相机 camera0 光心、轴向为 X 前/Y 左/Z 上。名义上将光心对应到采集模型
   三相机中间的 `joint_ca2` 原点，其平移为
   `[-0.000792, 0.000118586625, -0.0016148068] m`。
   该相机 link 的光学轴 X 右/Y 下/Z 前换为 raw 轴后，与采集模型 base 轴一致。
   因此还需从第 3 步坐标减去相机原点偏移，不能把模型 base 与 raw EE 直接视作同一点。

最终默认工具变换为：

```text
p_raw_eef = [-0.068277831741, 0.000372712242, -0.098051703299] m
R_raw_eef = Ry(-15°)
```

后端点来自上述配套法兰的名义转移，不是指尖中点，也不是手柄网格的极值点。
旋转沿用现有规范：接近方向沿手指固定伸出方向，掌背法向为旋转的 Z 列。
同一设备在所有 episode 中使用相同固定变换；更换指尖不改变该基准。

该偏移是模型推导的默认值，不是逐套实机标定。相机光心与模型 link 原点的名义对应、
两模型之间的几何差异及真实装配误差仍可能产生偏差；确认实机外参后可用
`--eef-offset X Y Z`、`--tool-rpy R P Y` 覆盖。无需为了转换而先完成逐套实机标定。
`camera_info` 的图像分辨率及外参问题仍应独立核对。

#### 复算

下载并解压上面两个附件后运行：

```bash
loom-data robot realomin-mount \
  --gripper-urdf /path/to/DAS_Gripper_V3/urdf/DAS_Gripper_V3.urdf \
  --controller-urdf /path/to/DAS_Controller_V3_with_flange/urdf/DAS_Controller_V3_with_flange.urdf
```

输出包含 raw EE 系中的后端偏移、采集模型 base 系中的后端点、驱动轴注册残差和两份
URDF 的 SHA-256。固定关节数值与附件哈希保存在
`tests/fixtures/realomin_v3_mount_geometry.json`，回归测试不需要下载模型。
转换运行时直接使用推导后的默认数值，无需加载 URDF、网格或访问网络。

实际夹爪值按 `openness = clip(value / 0.103, 0, 1)` 转换；
0 米闭合、0.103 米张开，采用数据集卡声明的范围，不按样本极值重标定。
非有限值报错；有限的微小越界（样本可略高于 0.103）裁剪。

指令默认取最近的非数字父目录技能名，将下划线换为空格；这属于源技能标签，
不是额外生成的逐帧语言标注。可用 `--instruction` 指定已有的任务文本。

### 发现与 episode ID

默认读取 `dataset/raw/10Kh-RealOmin-OpenData/`；完整数据在别处时用 `--input-dir` 指定。
有 `subset_manifest.json` 时只处理清单里的 MCAP；
完整数据目录没有该清单时递归枚举 MCAP。episode ID 是源相对路径的稳定 SHA-256 前 16 位，
不会因不同技能下有同名 `00001.mcap` 而发生碰撞。修改数据根的相对目录结构会改变 ID。
扫描排除隐藏缓存目录；`--limit` 按候选 episode 计数，在扫描过程中生效。

### 同步与视频

使用 MCAP 内嵌 Protobuf schema。只选 `robot0/1` 的中央相机 `camera0`，分别导出为 `left_wrist`、`right_wrist`。

帧序列覆盖四路 state 和两路相机都有数据的时间段。位姿做位置线性插值加旋转 slerp，夹爪线性插值，
RGB 取采集时刻最近的原生帧。帧时刻两侧的 state 样本间隔超过 500 ms 或最近 RGB 帧偏离超过 250 ms 的
episode 不导出，记为 `sampling_gap`；共同时间段不足两帧的记为 `no_common_interval`。

逐条 H.264 消息解码，校验消息与显示帧一一对应；消息数量、视频帧数不一致或显示帧对应时间递减时整集 drop。
RGB 以实际解码尺寸为准，不依赖旧 `camera_info` 尺寸：1600×1300（16:13）缩放为 640×520。

允许的布局只有 `left_wrist+right_wrist_640x520`：两路腕部相机齐全。缺一路的 episode 不导出，记为 `unsupported_layout`。

### 数据 drop

非有限 state、缺失四路 state、没有 RGB、无效 world 位姿、时间倒退、损坏 MCAP/视频及
视频消息与帧数/时间映射不一致均整集 drop。夹爪轻微越界仍裁剪。
Stage 2 的数字目录没有细粒度技能名，保留源场景名 `Clutter Tidy-Up [Stage2]` 作为指令，
不生成额外的技能标签。
