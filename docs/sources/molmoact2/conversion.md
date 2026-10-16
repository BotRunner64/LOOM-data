# MolmoAct2-BimanualYAM 转换

入口：`scripts/convert/molmoact2.py`。读取本地完整 LeRobot v3 数据集或本项目下载器生成的 episode 子集，
写出[统一导出结构](../../contract/export-contract.md)。字段依据见[源说明](source.md)。

```bash
pip install -e .
python scripts/convert/molmoact2.py
# 先转换一条；输出目录必须尚不存在
python scripts/convert/molmoact2.py --limit 1 --output-dir dataset/processed/MolmoAct2-preview
```

默认输入 `dataset/raw/MolmoAct2-BimanualYAM/`，默认输出
`dataset/processed/MolmoAct2-BimanualYAM/`。
`--input-dir` 可以指向原始数据根目录（包含 `meta/`、`data/`、`videos/`），
或下载器拆好的子集目录（包含 `source/meta/` 和 `episodes/`）；
`--model-dir` 默认 `assets/robot_models/yam/`，资产安装见项目 README。
也可运行 `python -m scripts.convert.molmoact2`。
完整数据从 `meta/episodes/**/*.parquet` 枚举 episode，按元数据中的 chunk/file 索引
定位帧表与三路拼接 AV1 视频，按各相机自己的时间区间解码并转成单集 H.264（CRF 18、fast）。
数据文件只缓存所需列，最多保留当前一个文件；视频逐路处理。完整数据建议使用
[并行转换入口](../../tools/parallel-conversion.md)，按 episode 分配 worker、提交与恢复。

## 转换规则

- 仅使用 `observation.state`；不读取 action，也不输出关节、速度等额外数据。
- 六个实测关节角按 rad 输入 YAM FK；左右臂分别以自身固定基座为参考系，
  不套用 ABC 的双臂安装外参。工具中心和轴向与 ABC 共用
  `scripts/robot/yam.py` 的实现，按源说明的固定变换生成位置加按列展平的完整旋转矩阵。
- 夹爪已有归一化：有限值裁剪到 `[0,1]`，范围内透传；NaN、无穷时报错。
- 直接将表内 `timestamp` 四舍五入成 int64 纳秒，减去共同首时刻。
  四路 state 和三相机索引沿用同一记录时间序列，不插值、补帧、排序或按 FPS 重建。
  **源仅保留 LeRobot 记录时间，无法恢复硬件采集时刻及各传感器异步节奏**；
  输出保留源所能提供的时间精度，不能视作独立硬件时间戳的测量结果。
- `top → top`、`left → left_wrist`、`right → right_wrist`。
  子集 MP4 必须为单路无音轨 H.264，逐帧解码确认分辨率稳定、帧数与表行数一致后按字节复制。
  原始拼接视频从前一个关键帧开始解码，仅保留 `[from_timestamp, to_timestamp)` 中的帧，
  校验区间长度、帧连续性和帧数后重新编码为 H.264，无需 GPU。
  容器 PTS 用于定位片段；输出采集时间仍使用帧表的原生 `timestamp`，不加拼接片段偏移。
- 逐 episode 有效详细标注优先，覆盖 `[0, 最后记录时刻]`；缺失、null、空白标注回退到
  标准任务表，按逐行 `task_index` 的文本变化生成不重叠区间。
  相同时间最后一次变化生效，末采样时刻的零长度区间不写出。
  以完整数据的 `meta/` 或子集的 `source/meta/` 任务表为准，不使用 `episode.json.task` 缓存覆盖它。

## 校验和发布

校验本体、版本、14 维状态字段名称及顺序、三路视频特征；
校验 episode 目录 ID、元数据长度、全局索引区间、逐行 episode ID、连续帧序和索引。
时间非有限、倒序、负值或整集零时长，状态维度错误、非有限数，任务索引未知、
任务表重复 ID、相机缺失或视频帧数不符均报错。

完整数据的 `prepare` 只建立计划；视频区间与帧数的不一致在对应 episode 转换时
报错，其他 episode 可继续提交，修复问题后可恢复。服务器 v1 的 episode 31749
帧表有 1,619 行，但 top/left 视频区间对应 2,832 帧、right 对应 2,658 帧；
转换器会明确拒绝这集，不猜测应删减或补齐哪些帧。`finalize` 默认跳过已记录错误的失败集，发布其余成功结果。

不覆盖已有输出。先在输出目录旁暂存，所有 episode 成功后才整体发布；
失败清理暂存，不留下已发布的半成品或缺少元数据的 episode。

```bash
python -m unittest discover -s tests -p 'test_convert_molmoact2.py'
python scripts/viewer/export_viewer.py dataset/processed/MolmoAct2-BimanualYAM
```

测试覆盖非零共同原点和不规则时间间隔、左右臂 FK/工具轴、夹爪裁剪、
标注优先和任务回退、视频复制与帧序、输入损坏及失败清理；YAM 几何测试需安装资产。
