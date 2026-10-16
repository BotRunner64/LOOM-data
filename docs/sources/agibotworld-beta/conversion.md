# AgiBot World Beta 转换

入口：`scripts/convert/agibotworld_beta.py`，遵循[目标格式契约](../../contract/export-contract.md)。
原始时钟、位姿参考系、工具安装与实测夹爪方向见[源契约](source.md)。

```bash
python scripts/convert/agibotworld_beta.py

python scripts/convert/agibotworld_beta.py --limit 1 \
  --output-dir dataset/processed/AgiBotWorld-Beta-preview
```

默认读取 `dataset/raw/AgiBotWorld-Beta/`，输出到 `dataset/processed/AgiBotWorld-Beta/`。
支持下载器生成的子集与保留 `proprio_stats/<task>/<episode>`、`observations/<task>/<episode>`、
`task_info/` 布局的完整解包数据；不直接扫描几十 GiB 的源 TAR。
有 `subset_manifest.json` 时只处理其中的 episode，忽略 `.archives/` 缓存和其它探查文件。
按任务逐个扫描；`--limit` 是候选 episode 数量，包含会被 drop 的 episode，在扫描时生效。
episode ID 为 `task_<task_id>_episode_<episode_id>`。

HDF5 的整数纳秒 `timestamp` 用作四路 state 与各路 RGB 的采集时刻，不按 MP4 PTS 或 FPS 重建。
源法兰已包含腰部运动，转换只组合实测底盘位姿及固定工具变换，不依赖机器人模型资产。
原 MP4 完整解码再写 H.264，不保留音轨，帧数必须与 HDF5 行数相同。
`action_config` 的指令帧区间映射到同一原始 clock；缺口使用整集任务文本。

依赖 `pyproject.toml` 中的 h5py、NumPy、PyArrow、PyAV、tqdm。

已知源数据异常按整集 drop：缺 state、RGB 或任务标注，冲突标注，灵巧手 `[N,12]`，
无效/非有限 state、时间倒退、无效位姿、视频损坏或帧数不符、重叠/越界的指令。
完全相同的重复标注去重并保留轨迹；冲突仅影响对应 episode。
`action/end` 的长度不参与导出校验，导出仍只使用实际 state。

底盘位置和四元数**同时全零**视为缺失标记。只有整集 `action/robot/velocity` 是
有限的 `[N,2]` 且绝对值不超过 `1e-6` 时才恢复：整集缺失使用固定底盘参考系；
部分缺失沿用之前最近的实测位姿，开头缺失使用首个有效位姿。
其余实测底盘位姿保留，不修改源 HDF5，不添加腰部 FK。
缺失位姿时速度命令非零、缺失或无效，位置非零但四元数全零，以及无效法兰姿态均 drop。

默认 TCP 是公开长指模型的几何估计，夹爪 `opened=35/closed=120` 是这批数据的暂定源读数基准。
两者都尚未取得实机标定；跨型号/固件需核对后覆盖参数。

- `--tcp-offset X Y Z`：TCP 在源法兰系的坐标，米。
- `--left-tool-rpy R P Y` / `--right-tool-rpy R P Y`：规范工具旋转相对源法兰，弧度。
- `--gripper-open VALUE` / `--gripper-closed VALUE`：实测源读数基准，单位与源一致，可反向。

单次 CLI 与并行入口共享 `prepare → worker → finalize`、逐集提交和自动 resume。
重复相同命令会跳过已导出和已 drop 的 episode；已发布且计划匹配的输出直接成功返回。
源变化、配置变化或旧策略计划不能直接 resume，需使用新输出目录。
并行入口为 `scripts/convert/parallel.py --dataset agibotworld_beta`，见[并行说明](../../tools/parallel-conversion.md)。

统计在工作目录的 `conversion_summary.json`：候选、导出、修复、drop、失败、未完成数量，
drop/修复占候选的比例及按原因计数。修复数是导出数的子集；多原因 drop 只算一个 episode。
`records/<episode_id>.json` 保存逐集 drop 原因/说明或修复原因；这些内容不写入目标 manifest。
工作目录规则见并行说明（普通输出为 `<output>.work`，仓库 `dataset/export` 对应 `dataset/work`）。
程序错误、依赖缺失和磁盘错误不作为数据 drop，worker 返回非零并保留错误记录。
所有 worker 退出后，可以修复后重跑 resume，或直接执行 finalize，省略有 episode 错误记录的
失败集并发布成功部分。没有错误记录的未完成集及恢复状态损坏仍阻止 finalize。
