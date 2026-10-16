# scripts 目录分类

脚本按用途分类，转换与检查分开，与项目 README 的组织原则对应：

```
scripts/
├── download/   # 源数据选取与下载（每个源数据集一个入口）
│   ├── abc_subset.py
│   ├── agibotworld_subset.py
│   ├── agibotworld_beta_subset.py
│   ├── galaxea_subset.py
│   ├── hifi_umi_subset.py
│   ├── molmoact2_subset.py
│   ├── realomin_subset.py
│   └── subset_common.py
├── convert/    # 转换到目标格式数据契约（每数据集一个入口 + 公共函数）
│   ├── abc130k.py
│   ├── agibotworld2026.py
│   ├── agibotworld_beta.py
│   ├── galaxea.py
│   ├── hifi_umi.py
│   ├── molmoact2.py
│   ├── realomin.py
│   ├── export_common.py
│   ├── export_validation.py
│   ├── source_policy.py
│   ├── parallel.py
│   └── parallel_adapters.py
├── robot/      # 机器人模型工具（URDF 解析、FK、可视化、包校验）
│   ├── urdf_model.py
│   ├── kinematics.py
│   ├── yam.py
│   ├── fk.py
│   ├── show_models.py
│   └── check_packages.py
├── check/      # 仅消费目标格式契约的自动检查工具
│   ├── anomaly_detection.py
│   ├── anomaly_stats.py
│   ├── parallel.py
│   └── dataset_stats.py
└── viewer/     # 仅消费目标格式契约的本地检查工具
```

- `download/`：只依赖各源数据集自身的契约，产出 `dataset/raw/` 与 `dataset/manifests/`。
- `convert/`：把源数据对齐到唯一的目标格式契约（见 [`../docs/contract/export-contract.md`](../docs/contract/export-contract.md)）。`export_common.py` 是跨数据集共用的 Parquet/视频写出逻辑；`parallel.py` 提供七个数据源的并行与自动 resume 入口，见[并行转换说明](../docs/tools/parallel-conversion.md)。
- `robot/`：面向成品 URDF 模型包的工具。`urdf_model.py` 与 `kinematics.py` 是库，`fk.py`、`show_models.py`、`check_packages.py` 是命令行入口；`convert/` 也复用前两者。
- `viewer/`：读取导出数据契约及可选异常记录的本地 viewer；异常界面只依赖记录格式，不依赖具体检测规则。不读取原始数据或机器人模型。用法见 [`../docs/tools/export-viewer.md`](../docs/tools/export-viewer.md)。
- `check/`：读取导出数据，按异常检测契约生成 JSONL 报告。用法见 [`../docs/tools/anomaly-detection.md`](../docs/tools/anomaly-detection.md)。
  `parallel.py` 提供异常检测的单机/Slurm 并行入口与自动 resume。
  数据集的 episode 数量和指令时间跨度快速汇总见 [`../docs/tools/dataset-stats.md`](../docs/tools/dataset-stats.md)。
  已有异常报告的汇总与按异常原因统计见 [`../docs/tools/anomaly-stats.md`](../docs/tools/anomaly-stats.md)。

## 运行方式

入口脚本支持两种方式，均以项目根目录为工作目录：

```bash
python scripts/convert/abc130k.py
python -m scripts.convert.abc130k      # 等价
```

`scripts/` 及其子目录是命名空间包（无 `__init__.py`），子包之间用
`from scripts.robot.kinematics import fk_poses` 之类的绝对导入；入口脚本在直接执行时会
自行把项目根目录加入 `sys.path`，因此两种方式都可用。

## RealOmin / AgiBot Beta 子集下载

使用已登录且有对应 gated 仓库访问权限的 Hugging Face 账号，默认各选 5 条完整轨迹。
manifest 固定到仓库 commit SHA；重复执行 `download` 可复用 Hub 缓存并恢复截断文件。

```bash
python scripts/download/realomin_subset.py select
python scripts/download/realomin_subset.py download

python scripts/download/agibotworld_beta_subset.py select
python scripts/download/agibotworld_beta_subset.py download
```

RealOmin 默认在 `Cooking_and_Kitchen_Clean/clean_bowl/00001` 内按 seed 42 抽样，
保留完整 MCAP 与 `DAS_Gripper_V3.zip`。可用 `--prefix`、`--strategy`、`--count` 调整选择。
输出：`dataset/raw/10Kh-RealOmin-OpenData/`；默认 manifest：
`dataset/manifests/realomin-clean-bowl-5.json`。

Beta 的 `select` 会先下载最小状态 TAR，再按大小检查视频 TAR 的实际成员，
以 `(task_id, episode_id)` 精确匹配状态和所有 RGB 视频。TAR 文件名中的起止编号不代表连续 episode。
`--max-download-gib` 默认 4，限制选择过程中下载的状态及视频分片总量；不足时明确报错。
可用 `--state-archive` 指定状态分片，用 `--count` 调整轨迹数量。
`download` 提取选中的 HDF5 与所有 RGB 视频，保留完整的上游任务 JSON。
输出：`dataset/raw/AgiBotWorld-Beta/`；默认 manifest：
`dataset/manifests/agibotworld-beta-5.json`。

Beta 中间分片保存在输出目录下的 `.archives/`，提取后不将 depth 或 parameters 放入子集。
本次选中 task 694 的 5 条轨迹，有效 HDF5/视频约 254 MiB，中间分片约 3.80 GiB。
转换入口为 `python scripts/convert/realomin.py` 与
`python scripts/convert/agibotworld_beta.py`，默认分别输出到
`dataset/processed/10Kh-RealOmin-OpenData/` 和 `dataset/processed/AgiBotWorld-Beta/`。
TCP 几何估计及夹爪标定限制见对应的源契约与转换说明。
