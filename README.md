# LOOM-data

把多个机器人数据集转换成同一种格式，并检查转换结果。

导出的是标准 LeRobot v3.0 数据集：所有数据同步到 30 fps，每帧包含四路实际 state（左右臂 EEF 位姿、左右夹爪开合）、RGB 图像和任务指令。格式由[导出契约](docs/contract/export-contract.md)唯一规定，所有数据源都向它对齐。

## 安装

```bash
conda env create -f environment.yml
conda activate loom-data
```

环境使用 Python 3.11，并以可编辑模式安装本项目。依赖有变化时运行 `pip install -e '.[viewer,model-preview]'` 同步。

部分数据源需要机器人模型，安装方法见[使用说明](docs/usage.md#机器人模型)。

## 用法

```bash
loom-data download hifi_umi select --count 5      # 选一个小样本
loom-data download hifi_umi download              # 下载到 dataset/raw/
loom-data convert run --dataset hifi_umi --workers 8   # 转换到 dataset/export/
loom-data check run dataset/export/HiFi-UMI-2K/top+left_wrist+right_wrist_640x480 --workers 8   # 检测异常
loom-data clean dataset/export/HiFi-UMI-2K/top+left_wrist+right_wrist_640x480   # 处理异常，原地清理
loom-data view dataset/export/HiFi-UMI-2K/top+left_wrist+right_wrist_640x480   # 在浏览器里查看
```

一个数据源按相机布局导出为一个或多个数据集，目录名即布局名（见导出契约第 2 节）。检测和查看的对象是其中一个数据集。

转换和检测中断后，重跑同一条命令即可续跑。Slurm 上的运行方式、存储预算采样、失败处理等见[使用说明](docs/usage.md)。

## 数据源

| 数据集 | 键 | 位姿来源 |
|---|---|---|
| [ABC-130K](docs/sources/abc130k.md) | `abc130k` | 关节角 + YAM 模型 FK |
| [MolmoAct2-BimanualYAM](docs/sources/molmoact2.md) | `molmoact2` | 关节角 + YAM 模型 FK |
| [HiFi-UMI-2K](docs/sources/hifi_umi.md) | `hifi_umi` | 源记录的手部位姿 |
| [Galaxea Open-World Dataset](docs/sources/galaxea.md) | `galaxea` | 源记录的末端位姿 + 躯干 FK |
| [AgiBot World 2026](docs/sources/agibotworld2026.md) | `agibotworld2026` | 源记录的法兰位姿 + 腰部 FK |
| [10Kh-RealOmin-OpenData](docs/sources/realomin.md) | `realomin` | 源记录的 VIO 位姿 |
| [AgiBot World Beta](docs/sources/agibotworld_beta.md) | `agibotworld_beta` | 源记录的法兰位姿 |

每个数据源的说明页记录它的原始格式，以及到导出契约的映射规则和依据。

## 文档

- [导出契约](docs/contract/export-contract.md)：导出数据包含什么、如何表示、如何存储。
- [异常检测契约](docs/contract/anomaly-contract.md)：异常检测的规则、阈值、报告格式和处理办法。
- [使用说明](docs/usage.md)：所有命令的用法。
- [开发说明](docs/development.md)：代码结构，以及如何新增数据源。
