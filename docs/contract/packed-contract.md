# 打包格式契约

本文规定打包数据的格式。打包数据是一个导出数据集的视频换一种存放方式：内容与[导出契约](export-contract.md)中的 MP4 逐字节相同，只是把每集每路相机一个小文件改成少量大文件，并给出每一帧的位置，供训练按帧随机读取。

打包数据只含视频。state、instruction 和 episode 元数据仍从导出数据集读取。导出数据集是唯一的权威来源，打包数据随时可以由它重新生成。

## 1. 存储结构

打包数据与导出数据分开存放，目录层级与导出数据相同：

```text
<打包根>/<导出名>/<源>/<布局>/       # 对应 <导出根>/<导出名>/<源>/<布局>/
├── info.json                         # 格式版本、相机顺序、分片清单
├── episodes.parquet                  # 每个 episode 一行，见第 3 节
├── frames.parquet                    # 每帧一行，见第 4 节
└── shards/000000.bin                 # 视频码流，见第 2 节
```

一个布局数据集对应一个打包目录。`info.json` 最后写入：目录里没有 `info.json` 说明打包没有完成，不能使用。

## 2. 分片

分片按 `episode_index` 顺序存放 episode 的视频码流，一个 episode 只在一个分片里。每个分片约 4 GiB，编号从 `000000` 起连续递增。

一个 episode 内按帧交错：先放第 0 帧各路相机的包，再放第 1 帧各路相机的包，依此类推。同一帧内各路相机的顺序就是 `info.json` 的 `cameras` 顺序。

每个包是导出 MP4 中对应帧的 H.264 包，按 MP4 中的原样存放（长度前缀格式），不重新编码。

## 3. episodes.parquet

| 列 | 类型 | 含义 |
|---|---|---|
| `episode_index` | int64 | 与导出数据集相同 |
| `episode_id` | string | 与导出数据集相同 |
| `length` | int64 | 帧数，与导出数据集相同 |
| `shard` | int64 | 所在分片的编号 |
| `<video_key>.extradata` | binary | 每路相机一列：该 episode 视频的 H.264 解码器配置（MP4 中的 `avcC`） |

`<video_key>` 是导出契约中相机的 feature 键，例如 `observation.images.top`。

**与导出数据的对应**：`episode_index`、`episode_id`、`length` 与导出数据集的 episode 元数据逐行一致。任何一行不一致，说明导出数据在打包后被改过，这份打包数据作废，需要重新打包。

## 4. frames.parquet

每帧一行，按 `episode_index`、`frame_index` 排序。

| 列 | 类型 | 含义 |
|---|---|---|
| `episode_index` | int64 | 所属 episode |
| `frame_index` | int64 | 帧号，从 0 开始 |
| `offset` | int64 | 该帧第一个包在分片中的字节位置 |
| `<video_key>.size` | int32 | 每路相机一列：该帧的包的字节数 |
| `<video_key>.keyframe` | bool | 每路相机一列：该帧是否为关键帧 |

同一帧各路相机的包首尾相接，第 `i` 路相机的包从 `offset` 加上前 `i` 路包的字节数处开始。

**解码一帧**：取该路相机在这一帧及之前最近的关键帧，用 `extradata` 初始化解码器，按帧号顺序送入从关键帧到目标帧的包，最后一个输出帧就是目标帧。导出视频的关键帧间隔至多 2 帧，所以最多读两帧的包。

## 5. info.json

| 键 | 值 |
|---|---|
| `format` | `"loom-packed"` |
| `version` | `1` |
| `codec` | `"h264"` |
| `fps` | `30` |
| `cameras` | 各路相机的 `<video_key>`，按 `top`、`left_wrist`、`right_wrist` 的顺序列出该数据集有的相机 |
| `episodes` | episode 数 |
| `frames` | 总帧数 |
| `shards` | 每个分片一项：`bytes`（字节数）和 `sha256`（内容的 SHA-256） |
