# 导出数据 viewer

本地 viewer 只读取[目标格式数据契约](../contract/export-contract.md)定义的导出目录。不读取原始数据集、转换脚本或机器人模型；增加数据源无需修改 viewer。

```bash
pip install -e '.[viewer]'
python scripts/viewer/export_viewer.py path/to/exported-dataset
```

打开终端给出的 `http://127.0.0.1:8765/`。可用 `--host`、`--port` 指定监听地址和端口。
也支持通过 VS Code / notebook 的端口代理访问（如 `https://host/.../proxy/8765/`）；
请保留 URL 末尾的 `/`，页面模块、接口和视频帧地址会相对该路径解析。

Episode 下拉框每页显示 100 条记录，按 manifest 顺序排列；使用 Previous / Next 翻页。
页面只请求当前页的 ID，并立即加载首个 episode，避免大数据集一次性创建大量下拉选项。
翻页保留当前播放的 episode；选择其他记录才切换。异常跳转到当前页之外的 episode 时，
下拉框会额外显示一条 Current 选项。
服务端启动时在系统临时目录构建磁盘 SQLite manifest 索引，只保留少量最近访问的记录；不会把
完整 `episodes.jsonl` 常驻 Python 内存，也不会向只读导出目录写 index/cache。进程退出时删除临时索引。

## 异常记录浏览

若导出目录下存在 `anomalies.jsonl`，viewer 自动加载；也可指定报告：

```bash
python scripts/viewer/export_viewer.py path/to/exported-dataset --anomalies /tmp/report.jsonl
```

异常界面仅依赖[异常记录格式](../contract/anomaly_detection.md#异常记录格式)的五个字段
`index / stream / start_ns / end_ns / reason`，不依赖检测实现、阈值或具体异常类型。
`reason` 原样显示，筛选项从报告动态生成；任何符合格式的新原因都无需修改 viewer。
`index` 按 `episodes.jsonl` 的物理行号（从 0 开始，空行计数）关联导出数据。
报告本身只含定位信息，相机和 state 仍从对应导出目录读取。

异常栏默认只显示当前记录序号、Previous / Next 和当前异常的流、reason、秒级时间及持续时长。
点击序号展开记录列表；点击 Filters 可按 episode、stream、reason 组合筛选，并可一键清空条件。
有筛选条件时入口显示条件数量，展开后可查看具体范围；序号的分母为筛选后的记录数。
原始 episode 标识、stream、精确区间等信息收在 Details 中。弹层可点击外部或按 Escape 收起。
通过记录列表或 Previous / Next 按报告顺序浏览。
选择记录会暂停播放，切换到对应 episode，并精确跳到 `start_ns`；此后可拖动或播放查看上下文。
当前 episode 的筛选结果按流标在共享时间轴下方，点击标记也可跳转。
区间采用 `[start_ns, end_ns)`，起止相等时显示为单点；所有时间统一显示为 2 位小数秒。
当前时间命中的流会在 state 信息中高亮，包括未被筛选显示的记录。
报告顺序和重复记录均保留，不推断原因含义或合并记录。

未提供报告且默认文件不存在时隐藏异常界面；空报告显示无异常。
显式指定的文件不存在或记录格式无效时启动报错；区间超过 episode 终点时在加载该 episode 时报错。
报告在启动时读取，更新后需重启 viewer。

## 数据播放

页面可选择 episode、拖动共享时间轴或以 1× 速度播放。当前指令显示在页面顶部，随时间轴更新。每路 state 和相机独立按自身的 `timestamp_ns` 取当前时间之前最近的样本；时间点早于首样本时留空，不做插值。焦点位于某个相机画面时，键盘左右方向键按该相机的原生 frame timestamp 单步；焦点位于 state curve 或其通道开关时，方向键按最近聚焦的 EEF/gripper 流原生 timestamp 单步。单步会暂停播放，不插值也不切换到固定 Hz 网格。相机按索引的 `frame_index` 解码 MP4 显示顺序中的帧，MP4 容器时间不参与采集时间计算。主画面按相机名的词根评分排序选取：俯视/全局视角（如 `top`、`top_head`、`overhead`）优先，头部/前视次之，腕部与鱼眼（如 `hand_left`、`*_wrist`、`*_fisheye`）靠后，缩略图也按此顺序排列；点击其他相机的缩略图即可切换。页面上方左侧显示相机主画面和横向缩略图，右侧显示 `left_eef / right_eef / left_gripper / right_gripper` 四路契约 state 曲线；曲线与共享时间轴同步，也可在曲线内点击或拖动跳转。默认显示左右 EEF 的 xyz 与两路 gripper，旋转 6D 分量可在曲线下方按需开启。下方并排显示左右臂的 3D TCP 当前位姿及短拖尾（窄屏时上下排列），固定保留之前 2 秒的原生样本。桌面布局按窗口剩余高度分配视频/曲线区和 3D state 区，让两者同屏可见；相机缩略图数量较多时可横向滚动。拖尾颜色从淡到亮指向当前位姿，不绘制更早或未来的轨迹。红 X、绿 Y、蓝 Z 为实体工具轴，白球中心为 TCP，白色实体夹指示意夹爪开合。坐标轴、小球与夹指均使用光照和深度遮挡，配合透视投影帮助分辨前后；这些几何体的尺寸仅作示意，不代表实际机器人外形或夹爪行程。左键拖动旋转，中键或右键拖动平移；滚轮以鼠标所在位置为中心缩放，重置视角恢复初始中心和观看方向。

3D 视图需要浏览器支持 WebGL 2。Three.js 固定版本随 viewer 本地提供，运行时无需访问外部 CDN；版本与许可证见 `scripts/viewer/vendor/README.md`。切换 episode 时释放旧视图的 GPU 资源。

左右臂各自处于契约规定的固定参考系；契约不保证两个参考系相同，也没有相机外参。3D 视图把各自的起始工具坐标系用作观看方向的基准：默认相机位于工具后上方（−X、+Z 一侧），向前观察，左右臂各略向内侧偏转，近似从头部向两手看的 ego 方向；重置视角恢复此方向。按本 episode 的位姿范围固定视角，使当前 TCP 点在画面中移动。这只是基于起始工具轴的观看方向约定，不是经过标定的头部相机视角。viewer 保留左右臂两个独立视图，不做跨臂空间拼接或视频上的 3D 叠加。画面与位姿通过共享时间轴对应。

逐帧图像由 PyAV 解码并通过 Pillow 缩放为接近页面显示尺寸的 JPEG；不同相机的请求可并行处理，每路 FFmpeg decoder 最多使用 4 个线程，避免多相机在大核数容器上创建数百线程。对本项目导出的 30 Hz MP4，远距离跳转从附近的关键帧开始解码；其他视频仍按显示顺序逐帧解码。拖动时间轴时同一相机只保留一个在途请求，完成后直接追到最新位置；大跨度或回退跳转会重建 decoder 释放原生缓存。播放仅等待主相机帧，缩略图不阻塞并且最多按 1 Hz 更新。viewer 仅在本机监听，面向人工检查单个导出目录。
