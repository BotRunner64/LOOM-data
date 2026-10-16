# Galaxea 转换

入口：`scripts/convert/galaxea.py`；支持 R1 Lite 和 R1 Pro，输出遵循[唯一导出契约](../../contract/export-contract.md)。字段、参考系和工具轴定义见[源数据契约](source.md)。

```bash
# 转换全部已解压任务中通过静止筛选的轨迹
python scripts/convert/galaxea.py

# 小样本；N 只计算成功接收的 episode，不把跳过的计入上限
python scripts/convert/galaxea.py --limit-episodes 2 \
  --output-dir dataset/processed/Galaxea-preview

# 指定本体；也可通过 --input-dir 指定单个已解压任务目录
python scripts/convert/galaxea.py --robot-type r1lite \
  --output-dir dataset/processed/Galaxea-r1lite
```

默认输入 `dataset/raw/Galaxea-Open-World-Dataset/`，输出 `dataset/processed/Galaxea-Open-World-Dataset/`。输入需要已解压的 LeRobot 目录；可以先运行 `scripts/download/galaxea_subset.py extract`。转换不重复处理同目录下的 tar.gz。`--limit N` 限制筛选本体后按目录排序的前 N 个任务；`--limit-episodes N` 限制接收的总集数。依赖项目的 NumPy、PyArrow、PyAV，以及 `assets/robot_models/galaxea_r1lite/`、`galaxea_r1pro/` 模型。

## 静止筛选

只检查整条 episode 的底盘速度命令（兼容三维 `[vx, vy, wz]` 和六维 twist）：每行均有限，且所有分量绝对值不超过 `1e-6`，即按浮点容差视为零。存在非零或无效命令的 episode 直接跳过，不截取静止片段。实测轮速和 IMU 不参与筛选，也不要求提供轮速字段。

不需要 `--stationary-base`，没有轮速阈值参数。没有跳过原因文件，控制台仅显示导出数和跳过数。源文件结构损坏、时序错误、错误本体或视频缺帧仍报错。

这是按速度命令选取轨迹，并假定其底盘固定；零命令不构成物理静止证明，刹停惯性、外力推动或滑动仍可能产生实际运动。

本地 244 集按此规则接收 213 集、跳过 31 集；其中 R1 Lite 193 集，R1 Pro 20 集。

## 映射与输出

- 保留源逐帧 `timestamp`，秒乘 `1e9` 后取整，所有 state 和 RGB 共享首样本原点，不由 FPS 推算时间。
- 实测原生末端位姿左乘躯干 FK，再右乘固定工具变换，输出底盘系下 TCP 的位置和完整 3×3 旋转矩阵（按列展平）。底盘只作为筛选后假定固定的参考系，不导出底盘或 action。
- G1 夹爪 `openness = clip(stroke_mm / 100, 0, 1)`；非有限值报错。
- 四路 RGB 逐帧转为无音轨 H.264，逐路检查解码帧数，索引保存源时间；视频容器使用 30 FPS，不代表采集频率。相机 ID 为 `head`、`head_right`、`left_wrist`、`right_wrist`。
- 指令按逐帧细粒度标签生成区间，缺失时回退到整体任务；不使用质量标签。重复时间戳的零长度区间不写入，保留该时刻最后生效的标签。
- episode ID 为 `<task_directory>_episode_<index:06d>`，避免不同任务的编号冲突。

默认 TCP 是模型指尖中点估计，未做实机标定。可分别用 `--r1lite-tcp-offset X Y Z`、`--r1pro-tcp-offset X Y Z` 指定相应原生 ee_pose 系下的标定位置（米）。

目标目录必须不存在。全部输出先写到同级暂存目录，成功后一次发布；出错则清理暂存，不覆盖已有导出。全部被筛除时输出空的 `episodes.jsonl` 和 `episodes/`，明确显示导出 0 集。

```bash
python -m unittest discover -s tests -p 'test_convert_galaxea.py'
python scripts/viewer/export_viewer.py dataset/processed/Galaxea-Open-World-Dataset
```
