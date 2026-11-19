# 使用说明

所有功能都在一个命令下：

```text
loom-data download       下载源数据集的小样本
loom-data convert        把源数据集转换为导出格式
loom-data check          在导出数据上检测异常
loom-data stats          统计导出数据集的 episode 数和时长
loom-data anomaly-stats  汇总异常报告
loom-data view           在浏览器里查看导出数据
loom-data robot          机器人模型工具
```

每个命令都支持 `--help`。数据集用键指定：`abc130k`、`molmoact2`、`hifi_umi`、`galaxea`、`agibotworld2026`、`realomin`、`agibotworld_beta`。

## 目录约定

旧格式（异步的逐 episode 目录加 `episodes.jsonl`）的导出不能续跑或原地升级，需要用新版转换器写到新的输出目录。续跑计划会核对导出契约版本。

默认路径都相对仓库根目录；设置环境变量 `LOOM_DATA_ROOT` 可以换一个根目录。

```text
dataset/manifests/         下载清单
dataset/raw/<数据集名>/     源数据
dataset/export/<数据集名>/<布局名>/  导出数据：每个相机布局一个 LeRobot v3.0 数据集
dataset/work/              转换和检测的续跑状态
assets/robot_models/       机器人模型
```

这些目录都不进 Git。任何命令都可以用 `--input-dir`、`--output-dir` 等参数指定别的位置。

## 下载

下载分两步：`select` 选出一个子集，存成固定到某个 commit 的清单；`download` 按清单下载。同一份清单可以重复下载出同样的数据。

```bash
loom-data download hifi_umi select --count 5
loom-data download hifi_umi download
```

- 清单默认存在 `dataset/manifests/`，数据默认下到 `dataset/raw/<数据集名>/`，用 `--manifest`、`--output-dir` 修改。
- 清单已存在时 `select` 会拒绝覆盖；要重新选择就换一个 `--manifest` 路径。
- `download` 中断后重跑同一条命令即可继续。失败的文件记录在输出目录的 `download_failures.json`。
- 各数据集的选择参数不同，见 `loom-data download <数据集> select --help`。
- Galaxea 下载的是任务归档，转换前还要运行 `loom-data download galaxea extract`。
- 需要登录才能访问的数据集，先运行 `hf auth login`。

完整数据集不通过这个命令下载。把它们放到 `dataset/raw/<数据集名>/` 或用 `--input-dir` 指定即可，转换支持的目录布局见各[数据源说明](sources/)。

## 转换

### 单机

```bash
loom-data convert run --dataset abc130k --workers 8
```

这一条命令依次完成准备计划、启动 8 个本地进程转换、核对并发布。输入默认 `dataset/raw/<数据集名>/`，输出默认 `dataset/export/<数据集名>/`。

- **中断后重跑同一条命令即可续跑**，已完成的 episode 不会重转。可以换 `--workers` 数量。
- 转换过程中用 `loom-data convert status --output-dir ...` 看进度：已完成、失败、未完成的数量，以及各机器正在处理的任务数。
- 导出目录只在全部完成后才出现；过程中的数据在工作目录里。
- 转换器自动按相机集合和分辨率把 episode 分到各布局的数据集里，例如
  `dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480/`。发布时打印每个布局的 episode 数和帧数。
- 每个 worker 的编解码都是单线程，提速靠增加 worker 数。
- 设置 `LOOM_DATA_NO_PROGRESS=1` 关闭进度条。

### 多机并行

调度由你控制，按顺序执行三步：

```bash
# 1. 准备计划，单进程运行一次
loom-data convert prepare --dataset abc130k \
  --input-dir dataset/raw/ABC-130K --output-dir dataset/export/ABC-130K

# 2. 在任意多台机器上启动 worker，每台都是同一条命令
loom-data convert worker --output-dir dataset/export/ABC-130K --processes auto

# 3. 所有 worker 退出后，核对并发布，单进程运行一次
loom-data convert finalize --output-dir dataset/export/ABC-130K
```

worker 之间没有固定分工，也不需要知道彼此的存在：每个 worker 在共享文件系统上认领一个任务，转完再认领下一个，没有可做的任务时自己退出。因此：

- **worker 可以随时增加。** 先启动的先干活，后加入的接着领剩下的任务。集群紧张时有多少空位就开多少。
- **worker 挂掉不需要人工处理。** 干活的 worker 每分钟更新一次自己的认领标记；标记超过 10 分钟没更新，其他还在运行的 worker 会接手那个任务。
- `--processes auto` 按容器的 CPU 限额决定本机开几个 worker 进程；也可以写具体数字，默认 1。
- 本轮失败的 episode 不会被别的 worker 反复重试，下一次 `prepare` 之后才重试。

各机器要能访问同一个 Python 环境、源数据和输出盘。Slurm 上用 `srun loom-data convert worker --output-dir ...` 一次拉起全部 worker；交互式实例上把第 2 步的命令作为启动后命令即可。

重新执行这三步就是续跑。**`prepare` 和 `finalize` 运行时不能有 worker 在跑**：`prepare` 会清掉上一轮遗留的认领标记和暂存文件。

### 选择要转换的数据

这些参数在第一次准备时固定，续跑时不能改：

| 参数 | 含义 |
|---|---|
| `--limit N` | 前 N 个 episode（AgiBot World 2026 是前 N 个归档） |
| `--limit-episodes N` | 全局前 N 个候选 episode，之后被跳过或 drop 的也占名额 |
| `--storage-budget SIZE` | 按预估体积采样，使导出大小接近 SIZE，例如 `100GiB` |
| `--sample-seed N` | 配合预算的采样种子，默认 0 |
| `--storage-estimate-ratio R` | 预估的导出/源文件体积比，默认 1 |

存储预算是近似目标：体积由源文件大小估算，不提前解码。被多个 episode 共用的文件（拼接视频、归档）按 episode 均摊。实际占用可能超出预算，请留余量；有小样本的实测结果时用 `--storage-estimate-ratio` 修正。选中的子集写入计划，换 worker 数不会改变它。

各数据源自己的参数（安装接口偏移、夹爪端点等）见[数据源说明](sources/)。这些参数同样固定在计划里。

### 结果不完整时

每个 episode 有三种结局：

| 结局 | 含义 | 记录位置 |
|---|---|---|
| 导出 | 转换成功并通过格式校验 | `records/<id>.json` 和导出目录 |
| 跳过 / drop | 契约或数据源规定不要的数据。所有源：相机集合不在允许布局内（`unsupported_layout`）、源采样有间隙（`sampling_gap`）、各流没有共同时间段（`no_common_interval`）。此外 Galaxea 的非静止轨迹；RealOmin、AgiBot Beta 的源数据异常 | `records/<id>.json`，含原因 |
| 失败 | 程序、依赖、磁盘错误，或其他数据源的坏数据 | `errors/<id>.err`，含 traceback |

有 episode 失败时命令返回非零，不发布。这时二选一：

- 修复问题后重跑，只会重试未完成的 episode；
- 接受现状：运行 `loom-data convert finalize --output-dir ...`，它会省略有错误记录的 episode，发布其余的。

没有错误记录的未完成 episode（例如 worker 被 kill）会阻止 `finalize`，需要先重跑。

改了源数据或转换参数后，换一个新的输出目录。工具会拒绝用不兼容的计划续跑。

### 工作目录

`dataset/export/<名称>` 的工作目录是 `dataset/work/<名称>.work`；其他位置的输出是 `<output>.work`。

```text
plan.json                  固定的配置、任务清单和输入文件指纹
claims/<任务序号>           正在处理该任务的 worker 的认领标记
records/<episode_id>.json  一个已导出或已跳过的 episode
errors/<episode_id>.err    未完成 episode 最近一次的失败信息
logs/worker-*.err          每个 worker 的启动和运行日志（成功也会有）
conversion_summary.json    导出、修复、drop、失败的数量和原因统计（有 drop 时写出）
dataset/                   已完成的 episode（每集一个暂存目录），发布后删除
publish/                   finalize 正在组装的各布局数据集，发布时整体改名为导出目录
published.json             发布标记：各布局的 episode 数和帧数
```

`finalize` 把暂存的 episode 按布局组装成 LeRobot v3.0 数据集：帧表合并写入 `data/`，每个 episode 的视频各占一个文件，从暂存目录硬链接过去，不重新编码。没有任何 episode 导出时 `finalize` 报错，不发布空数据集。

`errors/` 为空不代表没有问题：被 SIGKILL（包括 OOM）或节点故障打断的 worker 来不及写错误记录。这时看 `logs/` 里最后处理的任务，并用 `sacct -j JOB_ID --format=JobID,State,ExitCode,MaxRSS` 查作业状态。

发布后工作目录仍保留，用于核对重复运行。确认不再需要后可以手动删除。

## 检测异常

```bash
loom-data check run dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480 --workers 8
```

检测的对象是一个布局数据集。报告默认写到 `<数据集>/anomalies.jsonl`，用 `--output` 修改。规则、阈值和报告格式见[异常报告](anomaly-report.md)；阈值都可以用命令行参数覆盖，见 `loom-data check run --help`。

检测只读逐帧表里的 state，不解码视频。流程和转换一样：中断后重跑即可续跑；Slurm 上分三步执行：

```bash
loom-data check prepare dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480
srun loom-data check worker dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480
loom-data check finalize dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480
```

与转换不同的是，检测的 `finalize` 要求每个 episode 都成功，不会省略失败的。全部成功后才替换报告，失败时保留旧报告。

阈值固定在计划里。换阈值时用新的 `--output` 路径生成另一份报告。续跑状态在 `<报告路径>.anomaly-work`（导出目录下的报告对应 `dataset/work/` 里的同名路径）。

## 统计

```bash
loom-data stats                      # dataset/export 下的所有数据集
loom-data stats dataset/export/ABC-130K --json   # 该源的所有布局
loom-data anomaly-stats              # 汇总各数据集的 anomalies.jsonl
```

路径可以是一个布局数据集、一个源的目录或导出根目录；每个布局数据集单独一行，名称为 `<源>/<布局>`。两个命令都不打开逐帧表或视频：`stats` 只读 `meta/info.json`，`anomaly-stats` 读 episode 元数据和异常报告。

- `stats` 的时长是总帧数除以 30。
- `anomaly-stats` 的 `Records` 是报告行数；`Affected` 和各原因列按 episode 去重，原因之间可能重叠，不能相加。

## 查看

```bash
loom-data view dataset/export/ABC-130K/top+left_wrist+right_wrist_640x480
```

打开终端给出的 `http://127.0.0.1:8765/`。用 `--host`、`--port` 修改监听地址。通过 VS Code 或 notebook 的端口代理访问时，保留 URL 末尾的 `/`。

viewer 只读导出目录，不依赖源数据、转换代码或机器人模型。

- 页面上方是相机画面和四路 state 曲线，下方是左右臂 EEF 的 3D 位姿和 2 秒拖尾。它们共用一条时间轴，可以拖动或播放。
- 所有数据在同一条 30 fps 帧序列上，显示当前时刻所在的帧。焦点在相机画面或曲线上时，左右方向键按帧单步。
- 3D 视图里红、绿、蓝轴是工具的 X、Y、Z，白球是 EEF 原点（夹爪安装接口）。左右臂各在自己的参考系里显示，不做跨臂拼接；几何体尺寸只是示意。
- 导出目录下有 `anomalies.jsonl` 时自动加载，也可以用 `--anomalies` 指定报告。异常可以按 episode、数据流、原因筛选，点击后跳到对应时刻。报告在启动时读取，更新后要重启 viewer。

## 机器人模型

需要模型的数据源（ABC-130K、MolmoAct2、AgiBot World 2026、Galaxea）从 `assets/robot_models/` 读取 URDF。模型包不在 Git 里，从 ModelScope 的 [`BingqianWu/RobotDataHub-Assets`](https://modelscope.cn/models/BingqianWu/RobotDataHub-Assets) 下载后，把其中的 `assets/` 放到仓库根目录。

```bash
loom-data robot check                 # 校验模型包是否完整
loom-data robot fk --dataset ABC-130K --link arm_left_link6
loom-data robot fk --dataset Galaxea-Open-World-Dataset --robot-type r1pro --link arm_left_link7
loom-data robot show                  # 在 MuJoCo 里并排显示各模型，需要安装 model-preview 依赖
```
