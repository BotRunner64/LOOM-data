# robot_data_hub

- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)
- [Galaxea Open-World Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
- [HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)
- [MolmoAct2-BimanualYAM](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset)
- [10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData)
- [AgiBot World Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

## 环境安装

在项目根目录创建并激活 conda 环境：

```bash
conda env create -f environment.yml
conda activate loom-data
```

环境使用 Python 3.11，并以可编辑模式安装项目及 viewer、机器人模型预览依赖。
Python 依赖统一维护在 `pyproject.toml` 中。环境创建后，如需同步依赖，在项目根目录运行：

```bash
python -m pip install -e '.[viewer,model-preview]'
```

## 项目组织原则

数据清理工作拆成四类相互独立的产物，各司其职，互不混用：

1. **目标格式数据契约（唯一）**：定义清理后数据的统一格式与存储结构，整个项目只有一份，见 [`docs/contract/export-contract.md`](docs/contract/export-contract.md)。所有源数据都向它对齐，任何格式调整只改这一处。
2. **源数据集数据契约（每数据集一份）**：描述单个源数据集的原始字段、坐标系、时间戳等语义，每个数据集维护一份（如各数据集自带的格式说明），只讲“这个数据源长什么样”。
3. **转换代码（每数据集一份，共用函数）**：把每个源数据集转换成目标格式的脚本，一个数据集一个转换入口；跨数据集共用的逻辑（写 Parquet、编码视频、坐标变换等）下沉到共用函数，避免复制。
4. **目标格式 dataset viewer（人工检查用）**：读取目标格式数据并可视化，供人工核对转换结果是否与源数据语义一致；只依赖目标契约，不依赖任何单个源数据集。

这四点把“格式定义”与“实现”分开、“各数据源的差异”与“公共逻辑”分开、“转换”与“检查”分开。新增数据源时，只需新增一份源契约和一个转换脚本。

## 数据集

每个数据集对应一份源契约（原始字段、坐标系、时间戳）和一份转换说明（运行方式与映射规则）：

| 数据集 | 源数据契约 | 转换说明 |
|---|---|---|
| ABC-130k | [`docs/sources/abc130k/source.md`](docs/sources/abc130k/source.md) | [`docs/sources/abc130k/conversion.md`](docs/sources/abc130k/conversion.md) |
| AgiBot World 2026 | [`docs/sources/agibotworld2026/source.md`](docs/sources/agibotworld2026/source.md) | [`docs/sources/agibotworld2026/conversion.md`](docs/sources/agibotworld2026/conversion.md) |
| Galaxea Open-World Dataset | [`docs/sources/galaxea/source.md`](docs/sources/galaxea/source.md) | [`docs/sources/galaxea/conversion.md`](docs/sources/galaxea/conversion.md) |
| HiFi-UMI-2K | [`docs/sources/hifi-umi/source.md`](docs/sources/hifi-umi/source.md) | [`docs/sources/hifi-umi/conversion.md`](docs/sources/hifi-umi/conversion.md) |
| MolmoAct2-BimanualYAM | [`docs/sources/molmoact2/source.md`](docs/sources/molmoact2/source.md) | [`docs/sources/molmoact2/conversion.md`](docs/sources/molmoact2/conversion.md) |
| 10Kh-RealOmin-OpenData | [`docs/sources/realomin/source.md`](docs/sources/realomin/source.md) | [`docs/sources/realomin/conversion.md`](docs/sources/realomin/conversion.md) |
| AgiBot World Beta | [`docs/sources/agibotworld-beta/source.md`](docs/sources/agibotworld-beta/source.md) | [`docs/sources/agibotworld-beta/conversion.md`](docs/sources/agibotworld-beta/conversion.md) |

下载与选取脚本（每个源数据集一个入口）见 [`scripts/README.md`](scripts/README.md)。单机/Slurm 并行转换及自动 resume 见[并行转换说明](docs/tools/parallel-conversion.md)。

导出后的自动异常检测支持首尾静止、state 突变和中段静止，以及单机/Slurm 并行与自动 resume，运行方式见
[`docs/tools/anomaly-detection.md`](docs/tools/anomaly-detection.md)。

导出数据集的 episode 数量和指令时间跨度快速汇总见
[`docs/tools/dataset-stats.md`](docs/tools/dataset-stats.md)。
已有异常结果的终端汇总与按异常原因统计见
[`docs/tools/anomaly-stats.md`](docs/tools/anomaly-stats.md)。

## 统一机器人模型

ABC-130K / MolmoAct2 对应 YAM，AgiBotWorld2026 对应 G2，Galaxea 按源 `robot_type` 分别对应 R1 Lite 或 R1 Pro。
四套自包含 URDF（本体、双臂、夹爪、视觉/碰撞网格、材质和清单）的入口见
[`assets/robot_models/README.md`](assets/robot_models/README.md)。
`assets/` 不纳入 Git，完整资产公开保存在 ModelScope：
[`BingqianWu/RobotDataHub-Assets`](https://modelscope.cn/models/BingqianWu/RobotDataHub-Assets)。
下载该仓库后，将其中的 `assets/` 目录放回本项目根目录即可。
R1 Pro 本地资产位于 `assets/robot_models/galaxea_r1pro/`，来自官方 GalaxeaManipSim；已随其余资产同步到上述 ModelScope 仓库。
按本体选择的 FK 示例：

```bash
python scripts/robot/fk.py --dataset Galaxea-Open-World-Dataset --robot-type r1pro --link arm_left_link7
```

可直接用于可视化和 FK；数据集 TCP、零位及安装外参尚未逐帧标定，状态在清单中明确记录。
