# AUTO-OSU

[![release](https://img.shields.io/github/v/release/kanze1/AUTO-OSU?label=%E4%B8%8B%E8%BD%BD&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/releases)
[![stars](https://img.shields.io/github/stars/kanze1/AUTO-OSU?style=flat&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/stargazers)
[![tests](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml/badge.svg)](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT%20%2B%20%E7%BD%B2%E5%90%8D-4fb8ff)](LICENSE)

**丢进一首歌，一分钟后拿到一张能直接打的 osu!standard 或 osu!mania 4K 谱面。**

[English](README.en.md) · [下载](https://github.com/kanze1/AUTO-OSU/releases) · [怎么用](#怎么用) · [效果与局限](#效果与局限) · [原理](#原理) · [常见问题](#常见问题)

![AUTO-OSU 主界面](docs/screenshot_zh.png)

<details>
<summary>亮色主题 / 批量结果</summary>

![亮色主题](docs/screenshot_en.png)

![批量结果](docs/screenshot_busy.png)

</details>

## 为什么做这个

我打 osu! 很菜，但是很爱玩。最难受的事情是：想玩的歌没有人做图，自己又不会做图。
于是就有了 AUTO-OSU：把喜欢的歌丢进去，一分钟后就能开打。

现在是 v0，已经能出让我自己愿意打完整首的图：节奏卡在鼓点上，跳和串都是从十几万张 ranked 谱面里学来的，
四个难度一次出齐。接下来它会继续变强：谱师的设计意图、刻意的 highlight、更长的滑条、变速歌的多红线，都在路线图上。

如果你也是"就想用喜欢的歌打一把"的人，欢迎拿去用、提 issue、一起改。
觉得有用的话点个 ⭐ **Star**，对我很重要。—— kanzei

## 怎么用

### 免安装（Windows）

1. 到 [Releases](https://github.com/kanze1/AUTO-OSU/releases) 下载 `AUTO-OSU-<版本>-win64-cpu.zip`（模型已内置），解压到任意位置。
2. 双击 `AUTO-OSU.exe`。
3. 把歌拖进窗口，勾选难度，点 **生成谱面**。
4. `.osz` 写到 exe 旁边的 `output` 文件夹，并默认直接在 osu! 里打开（自动导入）。打开 osu! 就能在歌曲列表里找到。

游戏模式默认是 **osu!standard**，原有行为不变。选择 **osu!mania 4K（规则生成）** 会生成四轨键盘谱；它使用独立规则生成器，不会加载或冒充使用只以 standard 谱面训练的模型。

不需要显卡：3 分钟的歌、一个难度，在现代 CPU 上约 1 分钟（16 核实测 70 秒；RTX 4090 上 16 秒）。
Windows 10 / 11，64 位。

### 一键配置 GPU 加速

在窗口的「计算设备」下点击 **自动配置 GPU 加速**。程序会自动准备 uv、独立的 Python 3.12 和适合显卡驱动的 CUDA 版 PyTorch，完成真实 CUDA 运算检查后立即启用，无需重启窗口。

- 无需预装 Python、uv 或 CUDA Toolkit；需要 NVIDIA 显卡及驱动。
- 首次联网下载约数 GB；安装进度和详细日志直接显示在窗口中，可以取消。
- 环境保存在 `%LOCALAPPDATA%\AUTO-OSU\runtime`，不修改系统 Python。重新配置失败时保留原来可用的环境。
- `auto` 优先使用可用的 GPU；`cpu` 始终使用 CPU；显式选择 `cuda` 时，不可用会报出原因。

### 整个文件夹批量生成

切换 **文件夹批量**，选择或拖入目录，按需勾选「包含子文件夹」，再点击 **批量生成**。程序会扫描支持的音频和视频，并在队列中显示每首歌的状态。

每首歌保存到独立子目录，同名歌曲不会覆盖；坏文件记录错误后继续处理下一首。批次结束会生成 `batch-report-*.json`，包含成功输出、失败原因和运行设备。点击「停止后续任务」会完成当前歌曲，再停止剩余队列。批量模式由你选取生成的 `.osz` 导入 osu!。

### 支持什么音频

输入会先标准化再进流水线，所以格式基本不挑：

- 音频：mp3、ogg、wav、flac、m4a / aac、wma、opus、aiff、ape、alac 等；
- 视频：mp4、mkv、webm、mov、avi 等，自动抽出音轨；
- 打进 `.osz` 的音频保证 osu! 能播：mp3 和 ogg-vorbis 原样保留；其他格式转 mp3，码率跟着源走（无损源 320 kbps，有损源按原码率向上取整，192 kbps 起）。
- 源文件里的封面图会直接当谱面背景；视频源没有封面就截一帧。歌名、歌手也从标签里读。

程序自带 ffmpeg，不用另外安装。

### 界面

| 区域 | 说明 |
| --- | --- |
| 歌曲 | 单曲转换 / 文件夹批量，支持拖放；下方显示处理队列。 |
| 游戏模式 | 默认 osu!standard；也可选规则式 osu!mania 4K。 |
| 难度 | Easy / Normal / Hard / Insane 可多选，全部打进同一个 `.osz`。默认 Hard + Insane。 |
| 输出 | 保存目录；「生成后自动导入 osu!」会直接打开 `.osz`，等于双击它。 |
| 模型 | 显示模型是否就绪；缺失时一键下载（自动校验）。 |
| 计算设备 | 显示实际可用的 CUDA、显卡与显存；支持重新检测和一键配置 GPU 加速。 |
| 顶部封面 | 中 / 英切换，亮 / 暗切换。设置、上次的歌和目录都会记住。 |

**高级选项**（点「高级选项」展开）：

| 选项 | 说明 |
| --- | --- |
| 随机种子 | 同一首歌换个数字得到另一版摆放；同一个种子结果可复现。 |
| BPM / 偏移 | 留空自动检测；检测错了（常见于变速歌或前奏很空的歌）手动填。偏移单位 ms。 |
| 谱师名 | 写进 `.osu` 的 Creator，默认 AUTO-OSU。 |
| 星级条件 | 给模型的难度提示，留空按难度默认：Easy 2.0 / Normal 3.2 / Hard 4.5 / Insane 5.5。想让 Hard 更密就填大一点。 |
| 摆放质量 | 坐标模型的扩散步数：快速 50 / 标准 100 / 精细 200。标准档够用。 |
| 生成引擎 | 「AI 模型」是正常模式；「纯规则」不用模型、几秒出图，只在没模型或想对比时用。 |
| 试听 mp3 | 另存一个原曲压低音量、每个物件加点击声的 mp3，不开 osu! 也能听节奏对不对。 |

### 命令行

`python -m autoosu 歌曲 [选项]`，exe 也认同样的参数（`AUTO-OSU.exe 歌曲.mp3 -d Hard`）。

```powershell
python -m autoosu --setup-runtime
python -m autoosu --check-cuda
python -m autoosu "D:\Music" --recursive -d Hard Insane -o "D:\Beatmaps"
python -m autoosu "D:\Music\song.mp3" --mode mania4k -d Hard --seed 42 -o "D:\Beatmaps"
```

| 选项 | 说明 |
| --- | --- |
| `--mode standard\|mania4k` | 游戏模式，默认 `standard`；`mania4k` 固定使用规则引擎 |
| `-d Easy Normal Hard Insane` | 要生成的难度 |
| `-o 目录` | 输出目录，默认 `out` |
| `--seed N` | 随机种子 |
| `--bpm` / `--offset` | 手动 BPM / 红线偏移（ms） |
| `--title` / `--artist` / `--creator` | 覆盖元数据（默认读音频标签，或「歌手 - 歌名」文件名） |
| `--star X` | 星级条件 |
| `--coord-steps N` | 扩散步数，默认 100 |
| `--cfg-scale X` | 坐标模型的 classifier-free guidance，默认 1.0 |
| `--temperature` / `--density` / `--density-bias` / `--decode-steps` | 节奏模型采样参数：温度、目标每小节物件数、"不放"偏置（负数更密）、解码轮数 |
| `--device auto\|cuda\|cpu` | 计算设备 |
| `--setup-runtime` | 使用 uv 自动安装并验证应用专属 GPU 环境 |
| `--check-cuda` | 检查当前实际推理环境（包括自动安装的环境） |
| `--recursive` | 输入为目录时，包含子文件夹 |
| `--rules` | 纯规则模式 |
| `--rhythm-model` / `--coord-model` | 指定模型文件；不指定则在 `models/` 里找 |
| `--no-coord-model` | 只用节奏模型，摆放走规则 |
| `--download` | 缺模型时从 GitHub Release 下载 |
| `--osu-shift 26` | 写入 `.osu` 时物件提前多少 ms |
| `--preview` | 另存试听 mp3 |
| `--debug-plot` | 另存分析图：响度与 kiai 段、onset 与拍线、各难度选中的音符 |
| `--dump-events` | 打印每个物件的时间、类型、拍位 |

`mania4k` 由规则生成并固定在 CPU 上运行。显式传入 `--rules` 或模型专用参数（包括 `--device`）会报错并退出，不会生成文件；这些参数只适用于 standard。

### Python 安装

```bash
git clone https://github.com/kanze1/AUTO-OSU
cd AUTO-OSU
python -m venv .venv && .venv\Scripts\activate       # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -e .[gui]
python -m autoosu --setup-runtime                     # 可选：自动准备独立 GPU 环境
python -m autoosu --download                          # 第一次下载模型（约 320 MB）
python -m autoosu                                     # 打开图形界面
python -m autoosu "歌曲.mp3" -d Hard Insane -o out    # 命令行
```

Python 3.10 及以上。macOS / Linux 用这种方式运行，exe 只提供 Windows 版。

## 效果与局限

- **节奏贴鼓点。** 节奏模型在验证集上与人类谱面的 onset F1 达到 0.96；作为参照，同一首歌两个人类难度之间只有 0.74。
- **摆放像人写的。** 坐标模型从纯噪声生成坐标，跳、串、滑条形状都是学来的；每条滑条都经过贴合检查，不会出屏幕。
- **还差的地方。** 一条红线；滑条长度不是模型输入，快歌上的长滑条偶尔被缩短（会补绿线保证时长正确）；
  打击音效只有基于鼓的简单 whistle / clap / finish；没有 storyboard。投稿之前请在编辑器里过一遍。
- **mania 4K 是诚实的规则式首版。** 它把检测到的节奏放入四个固定轨道，根据难度控制密度、和弦和长按，并阻止同轨长按与后续物件重叠；随机种子可复现布局。它没有使用或训练 mania 模型，也不声称达到人工谱师质量。当前仍是一条红线，轨道编排不理解指法流派；请在 osu! 编辑器中检查 timing、可读性和手感后再分享。

## 原理

![架构图](docs/architecture.png)

**分析与定时（规则）。** 打击 / 旋律分离，分频段（底鼓 / 军鼓 / 镲）算 onset 包络；tempogram 估 BPM 并纠正倍频；
在波形上做 1 ms 精度的偏移校准；用底鼓、和声变化和响度找小节起拍；按响度分段找 kiai。
物件统一比音频瞬态提前 26 ms 写入，这是 ranked 谱面的普遍惯例，玩家的偏移设置都按它校准。

**mania 4K（规则）。** 复用相同的音频分析、定时、元数据、封面和 `.osz` 打包；每个难度从节奏网格筛选音符，再以 seeded 随机分配四轨，优先换手并减少连续同轨。强拍可生成受控的双押/三押，持续声音可生成量化长按；任何仍被长按占用的轨道都不会接收新物件。输出明确写入 `Mode:3` 和 `CircleSize:4`。

**节奏模型** `rhythm_v0.pt`，约 2900 万参数。1/4 拍网格上的双向 Transformer：每个 tick 看 ±80 ms 的梅尔频谱、
在小节里的位置、局部响度，加上要求的星级 / CS / AR / OD / HP，预测六类之一（无 / 圈 / 滑条头 / 身 / 尾 / 转盘），
MaskGIT 式 12 轮并行解码。

**坐标模型** `coord_v0.pt`，约 1.3 亿参数。结构是 [osu-diffusion](https://github.com/OliBomby/osu-diffusion) 的 DiT-B，
本项目用完整的 1000 步噪声计划从零训练。输入是物件 token 序列（圈、滑条头、锚点、滑条尾、转盘，各带时间），
从纯噪声去噪出每个点的 x / y；条件只有星级和 CS，训练时一半样本把"到上一个点的距离"置零，所以它会自己决定跨度。

**滑条贴合。** 模型画形状，长度由节奏决定：每条滑条围绕滑条头缩放到要求长度；会出场地就先镜像、再旋转，
实在放不下才缩短并补一条本地绿线。快歌会按 BPM 压低 SliderMultiplier。

### 训练记录

![训练曲线](docs/training_curves.png)

| 模型 | 数据 | 硬件 | 步数 | 训练时长 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 节奏 masked v0 | 139,582 张 osu!standard 谱面（[project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps)） | 2 × RTX 5880 Ada | 60k，batch 128 | 4.5 h | 取第 40k 步：生成 onset F1 0.963，密度误差 0.10 |
| 坐标 DiT-B v0 | 同一语料的 140,018 张（ORS 布局） | 2 × RTX 5880 Ada | 200k，batch 128 | 8.5 h | 最终 loss 0.125；出界 0%，同种子 5 万 / 10 万 / 20 万步布局几乎一致 |

自回归版本的节奏模型也训过，因果注意力听不到后面的音频、还爱用转盘逃课，被掩码版淘汰了。
完整实验记录（含失败的路线）在 [docs/rhythm_model_design.md](docs/rhythm_model_design.md)，
wandb 项目：[autoosu-rhythm](https://wandb.ai/kanzei/autoosu-rhythm)、[autoosu-coords](https://wandb.ai/kanzei/autoosu-coords)。

## 自己训练 / 打包

训练用到的全部代码都在仓库里：`autoosu/ml/prepare_data.py`（HF 分片 → 特征与标签）、`autoosu/ml/train.py`（节奏模型，`torchrun` 多卡）、
`coord/` + `scripts/coord_make_ors.py`（坐标模型，accelerate）、`scripts/server_*.sh`（服务器流程）、`scripts/coord_export.py`（导出发布文件）。
两个模型文件都能用 `torch.load(weights_only=True)` 加载，不含 pickle 代码。

打 exe：

```powershell
pip install -e .[build]
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # dist/AUTO-OSU-<版本>-win64-cpu.zip
```

用装了 CUDA 版 torch 的环境加 `-Venv .venv-gpu -Suffix cuda` 可打显卡版。`python scripts/make_icon.py 头像.png` 生成窗口头像和 exe 图标。

## 常见问题

**模型下载失败？** 手动从 [models-v0](https://github.com/kanze1/AUTO-OSU/releases/tag/models-v0) 下载 `rhythm_v0.pt` 和 `coord_v0.pt`，
放到 exe 旁边的 `models/` 文件夹（或 `~/.autoosu/models/`）。

**杀毒软件报毒？** PyInstaller 打的包常被误报。可以用 Python 方式运行，或自己按上面的步骤打包。

**BPM 或偏移不对？** 在高级选项里手动填。变速歌目前只会给一条红线。

**osu! 没有自动打开？** 说明 `.osz` 没有关联到 osu!，把生成的 `.osz` 拖进 osu! 窗口即可。

**生成很慢？** CPU 一个难度约一分钟属于正常；摆放质量选「快速」能快一倍；有 NVIDIA 显卡可在窗口点击「自动配置 GPU 加速」。

**某个格式解不开？** 先确认文件本身能播放；程序会先用 libsndfile 再用自带的 ffmpeg 解码，都失败会提示具体原因。

## 路线图

- 坐标模型 v1：把滑条要求长度作为逐点条件，根治长滑条缩短。
- 节奏模型：谱师风格条件、1/12 网格覆盖三连音。
- 同一首歌多种子出多版供挑选；变速歌多红线。

## 许可与署名

代码和模型采用 **MIT 协议加一条署名条款**（见 [LICENSE](LICENSE)）：

- 个人使用、学习、社区里随便玩：只要保留版权声明即可，和普通 MIT 一样。
- **商业使用或大规模部署**（比如公开的网页服务、面向大众分发的应用、批量为平台或社区生成谱面）：
  必须在产品界面、关于页、商店页或文档的显眼位置注明 **AUTO-OSU by kanzei** 并附上本仓库链接 https://github.com/kanze1/AUTO-OSU 。

生成的谱面归你，歌曲归原作者。

## 致谢

- [osu-diffusion](https://github.com/OliBomby/osu-diffusion)（MIT）—— DiT 结构与扩散代码，收录在 `autoosu/ml/coord`。
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)（MIT）—— token 化思路和对比基线。
- [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) —— 训练语料。
- [osu-dreamer](https://github.com/jaswon/osu-dreamer) —— 早期基线。
- [Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC)（OFL）—— 界面字体，以 AUTO-OSU Sans 之名随程序打包。

作者：kanzei
