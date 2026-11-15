# 开发说明

## 代码结构

```text
src/loom_data/
├── cli.py            命令入口：loom-data <command>
├── paths.py          默认目录
├── contract.py       导出契约的代码形式：文件名、schema、写入、校验
├── video.py          视频解码、转码、切段
├── geometry.py       位姿和工具变换
├── resumable.py      可续跑作业的公共部分：原子写入、输入指纹、worker 启动与日志
├── sources/          每个数据源一个模块，实现统一接口
├── convert/          转换作业：配置、计划、worker、发布
├── check/            异常规则、检测作业、统计
├── download/         从 Hugging Face 下载子集
├── robot/            URDF、正运动学、模型工具
└── viewer/           查看导出数据的本地网页
```

依赖方向是单向的：

- `contract.py` 是唯一了解导出格式的地方。转换通过它写入和校验；检测和 viewer 从它取文件名和列名。
- `sources/` 只负责「这个数据源的一个 episode 怎么变成契约格式」，不关心并行、续跑和发布。
- `convert/job.py` 只负责计划、续跑和发布，不包含任何数据源的知识。
- `check/` 和 `viewer/` 只读导出数据，不依赖 `sources/`。

[导出契约](contract/export-contract.md)是项目的核心，格式的任何调整都先改那份文档，再改 `contract.py`。

## 可续跑作业

转换和异常检测都是同一种作业：`prepare` → 多个 `worker` → `finalize`。

1. **prepare** 发现所有任务，写成固定的计划。计划记录配置、任务顺序和输入文件的大小与修改时间。续跑时读取已有计划，不重新发现任务。
2. **worker** 按 `任务序号 % worker 数 == 自己的编号` 领取任务，跳过已完成的 episode。因为分配只取决于固定的任务序号，换 worker 数不影响结果。
3. **finalize** 核对计划中的每个 episode，生成 `episodes.jsonl`（或异常报告），然后把结果原子地改名到最终位置。

转换时一个 episode 的提交顺序是：

1. 在 worker 自己的暂存目录里写完，并通过 `contract.validate_episode` 校验；
2. 把 `records/<id>.json` 原子地写入工作目录；
3. 把 episode 目录原子地改名到 `dataset/episodes/<id>`。**这一步是提交点。**

有效的 record 和 episode 目录同时存在、且文件清单和大小一致，才算已提交。只有 record 没有目录，说明在第 2、3 步之间中断，重跑时会重新转换。目录存在却没有 record、record 损坏等矛盾状态会报错停止，不会被静默跳过或覆盖。

调度单位随数据源的读取方式而定：多数源一个任务是一个 episode；HiFi-UMI 完整数据是一个 shard，AgiBot World 2026 是一个归档。提交单位始终是 episode。

工作目录、暂存目录和最终输出要在同一个支持原子 rename 的文件系统上。

## 新增数据源

1. 在 `sources/__init__.py` 的 `TITLES` 里加一项：数据集键和上游名称。
2. 新建 `sources/<键>.py`，定义一个 `Source` 子类并导出为 `SOURCE`：

   ```python
   class MySource(Source):
       name = 'my_source'
       model = None            # 需要机器人模型时填 assets/robot_models 下的目录名
       policy = 'strict'       # 'skip'：episode() 可返回 None；'drop'：SourceDataError 记为 drop
       OPTIONS = {             # 本数据源的命令行选项
           'eef_offset': dict(type=float, nargs=3, help='...'),
       }

       @classmethod
       def make_params(cls, options):
           """校验选项并返回写入计划的参数。"""

       @classmethod
       def discover(cls, config, progress):
           """返回 (任务列表, 共享的输入文件)。每个任务含 source、episode_ids、inputs。"""

       def episode(self, task, destination):
           """把一个 episode 写到 destination，返回它的 episodes.jsonl 记录。"""
   ```

   `episode()` 用 `contract.write_state`、`contract.write_video_index` 和 `video.py` 里的函数写文件。一个任务包含多个 episode 时改为重写 `process()`，参考 `sources/agibotworld2026.py`。

3. 需要下载子集时，新建 `download/<键>.py`，提供 `REPO_ID`、`MANIFEST`、`WORKERS`、`COMMANDS` 和 `add_arguments`，参考 `download/realomin.py`。
4. 在 `docs/sources/` 写一页说明：原始格式，以及到契约的映射规则。契约要求每个源记录它的工具变换和安装接口原点。

作业、检测、统计和 viewer 都不用改。

## 测试

```bash
python -m unittest discover -s tests      # Python
node --test tests/*.mjs                   # viewer 的 JavaScript
```

- 需要机器人模型的测试在缺少 `assets/robot_models/` 时会失败或跳过，先按[使用说明](usage.md#机器人模型)安装模型。
- `tests/support.py` 提供转换和检测的测试辅助函数；`tests/fixtures/` 是从真实数据抽取的位姿样本，用于回归检查坐标系解释。
