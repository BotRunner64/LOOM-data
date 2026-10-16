# RealOmin 转换

入口：`scripts/convert/realomin.py`，遵循[目标格式契约](../../contract/export-contract.md)。
原始字段、固定工具变换的依据及标定限制见[源契约](source.md)。

```bash
python scripts/convert/realomin.py

python scripts/convert/realomin.py --limit 1 \
  --output-dir dataset/processed/RealOmin-preview
```

优先读取服务器目录 `dataset/raw/genrobot-realomin-10kh/`；该目录不存在时默认读取
子集目录 `dataset/raw/10Kh-RealOmin-OpenData/`。输出到
`dataset/processed/10Kh-RealOmin-OpenData/`。有 `subset_manifest.json` 时只处理清单里的 MCAP；
完整数据目录没有该清单时递归枚举 MCAP。episode ID 是源相对路径的稳定 SHA-256 前 16 位，
不会因不同技能下有同名 `00001.mcap` 而发生碰撞。修改数据根的相对目录结构会改变 ID。
扫描排除隐藏缓存目录；`--limit` 按候选 episode 计数，在扫描过程中生效。

使用 MCAP 内嵌 Protobuf schema；双臂位姿、磁编码器和全部 `robot0/1` 相机消息按原生时刻分别输出。
逐条 H.264 消息解码并统一编码无音轨 MP4，保留消息与显示帧的一一对应关系；
消息数量、视频帧数不一致或显示帧对应时间递减时报错。RGB 使用实际解码尺寸，不依赖旧 `camera_info` 尺寸。

需要 PyAV、MCAP、mcap-protobuf-support、NumPy、PyArrow（均列于 `pyproject.toml`）。
运行时不依赖资产目录，默认工具变换由本次 V3 模型几何固化；
`--tcp-offset X Y Z`（米）与 `--tool-rpy R P Y`（弧度）可覆盖原生工具到规范 TCP 的固定变换。
默认 TCP 缺少 VIO 与实机工具间的精确外参，不能把几何估计视为实机标定。

夹爪默认按 0–0.103 米归一化，可用 `--gripper-closed` / `--gripper-open` 覆盖声明范围。
`--instruction TEXT` 可覆盖从路径技能名称取得的任务文本。
非有限 state、缺失四路 state、没有 RGB、无效 world 位姿、时间倒退、损坏 MCAP/视频及
视频消息与帧数/时间映射不一致均整集 drop。夹爪轻微越界仍裁剪。
Stage 2 的数字目录没有细粒度技能名，保留源场景名 `Clutter Tidy-Up [Stage2]` 作为指令，
不生成额外的技能标签。

单次 CLI 与 `scripts/convert/parallel.py --dataset realomin` 共享逐集提交和自动 resume，
见[并行说明](../../tools/parallel-conversion.md)。已导出和已 drop 的 episode 重跑时跳过；
已有且计划匹配的最终输出直接成功返回。配置、策略或源变化时使用新输出目录。
工作目录保存 `conversion_summary.json`（导出、修复、drop、失败、未完成数量、比例及原因计数）
和 `records/<episode_id>.json`（逐集原因/说明）；不在目标 manifest 加入诊断字段。
程序、依赖和磁盘错误使 worker 返回非零并保留错误记录。所有 worker 退出后，可修复后
重跑，或直接 finalize，省略有 episode 错误记录的失败集并发布成功部分。
没有错误记录的未完成集及恢复状态损坏仍阻止 finalize。
