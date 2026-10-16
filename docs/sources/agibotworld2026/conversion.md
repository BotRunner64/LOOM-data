# AgiBot World 2026 转换

入口：`scripts/convert/agibotworld2026.py`。输出遵循[目标契约](../../contract/export-contract.md)；源字段解释见[源数据契约](source.md)。

```bash
# 转换前 3 条轨迹到 processed；输出路径须尚不存在
python scripts/convert/agibotworld2026.py \
  --limit-episodes 3 \
  --output-dir dataset/processed/AgiBotWorld2026-preview

# 全部已下载归档
python scripts/convert/agibotworld2026.py
```

默认递归读取 `dataset/raw/AgiBotWorld2026/` 下全部 `.tar.gz`，输出到 `dataset/processed/AgiBotWorld2026/`。`--limit-episodes N` 限制跨归档的前 N 条轨迹；`--limit N` 限制前 N 个排序后的归档。`--model-dir` 默认 `assets/robot_models/agibot_g2/`。需要 `pyproject.toml` 中的 PyAV、PyArrow、NumPy 依赖以及 G2 模型资产。

## 位姿转换与已知限制

双臂使用源记录的末端位姿；G2 URDF 只对腰部 5 关节做 FK，将随躯干的 `arm_base_link` 变换到底盘参考系。有底盘位姿时再左乘源底盘变换。缺失底盘位姿的 `task_3401/399093_399454.tar.gz` 按静止底盘约定处理，并逐集核验底盘速度命令为零。

工具轴、TCP 和开合归一化详见[源数据契约](source.md)。**默认 TCP 为长指夹爪几何估计 `[0, 0, 0.207056] m`，尚无该批实机标定**；可用 `--tcp-offset X Y Z` 提供源法兰坐标下的标定值。预览尚不能作为精确 TCP 的标定结果。

### 2026-09-23 预览修正

旧版有三项已确认的错误：用不匹配的 `crsB` 手臂模型重算末端；工具旋转矩阵的列与声明的轴向不符；夹爪开合取反。旧 TCP 还误用了 OmniPicker 中间 link 的原点，并非实机抓取中心。已改为源末端加腰部变换、规范轴映射和视频核对后的开合方向，TCP 改为上述显式估计。已有旧导出不会被脚本自动覆盖，需要重新生成；本地 `AgiBotWorld2026-preview` 的前三集 state 已按修正版本更新。

回归测试使用第 1 集的 6 个真实源采样，检查腕部相机相对导出工具的刚性关系；另检查工具轴、移动底盘组合及开合方向。刚性检查无法验证固定 TCP 偏移本身。

### 为什么初始视角对齐后仍可能与视频不同

导出 state 在固定参考系，`top_head` 却是移动相机。第 1 集源外参每 10 帧抽样显示，相机相对首帧最大平移约 `0.47 m`、旋转约 `91°`。viewer 手动调整一次视角只能对齐某一时刻；严格图像对齐需要每帧的相机外参和内参。目标契约不包含这些标定，当前 viewer 也未按源相机运动驱动观察视角。不能将 state 改成随动头部系来消除这一差异，那会违反固定参考系要求。

所有 state 与 RGB 均使用源 Parquet 的 `timestamp` 列，转为纳秒并共享本集共同零点，不按视频 FPS 推算。每路 RGB 视频逐帧解码后用 libx264 编成无音轨 H.264；深度视频不导出。源视频每集每路必须与 Parquet 行数一致。`default` 指令分段按源帧边界映射到源时间戳；缺口用整集任务文本填充。

输出在目标目录旁暂存，全部成功后才发布，不覆盖已有目录。大归档转换会多次顺序读取压缩包；暂存目录需要容纳本次输出和单路源 MP4 的空间。
