# AUTO-OSU

[![release](https://img.shields.io/github/v/release/kanze1/AUTO-OSU?label=%E4%B8%8B%E8%BD%BD&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/releases)
[![tests](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml/badge.svg)](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml)
[![license](https://img.shields.io/github/license/kanze1/AUTO-OSU?color=4fb8ff)](LICENSE)

**丢进一首歌，一分钟后拿到一张能直接打的 osu!standard 谱面。**

AUTO-OSU 用两个在 14 万张 ranked 谱面上训练的模型自动作图：一个决定**什么时候**放圈、滑条、转盘，
一个决定**放在哪里**。生成的 `.osz` 会自动导入 osu!，Easy 到 Insane 四个难度一次出齐。

[English](README.en.md) · [下载](https://github.com/kanze1/AUTO-OSU/releases) · [快速开始](#快速开始) · [原理](#原理) · [常见问题](#常见问题)

![AUTO-OSU 主界面（暗色）](docs/screenshot_zh.png)

<details>
<summary>亮色主题 / 生成中的顶部脉冲</summary>

![亮色主题](docs/screenshot_en.png)

![生成中](docs/screenshot_busy.png)

</details>

## 效果如何

- **节奏贴鼓点。** 节奏模型在验证集上与人类谱面的 onset F1 达到 0.96；作为参照，同一首歌两个人类难度之间只有 0.74。
- **摆放像人写的。** 坐标模型从纯噪声生成物件坐标，跳、串、滑条形状都是学来的，不是规则堆出来的；每条滑条都经过贴合检查，不会出屏幕。
- **诚实地说，这是 v0。** 一首歌只有一条红线；滑条长度还不是模型的输入，快歌上的长滑条偶尔会被缩短；打击音效只有基于鼓的简单 whistle / clap / finish。投稿之前请在编辑器里过一遍。

## 快速开始

### 免安装（Windows）

1. 到 [Releases](https://github.com/kanze1/AUTO-OSU/releases) 下载 `AUTO-OSU-<版本>-win64-cpu.zip`（约 550 MB，模型已内置），解压到任意位置。
2. 双击 `AUTO-OSU.exe`。
3. 把歌拖进窗口（支持 mp3 / ogg / wav / flac / m4a），勾选难度，点 **生成谱面**。
4. `.osz` 写到 exe 旁边的 `output` 文件夹，并默认直接在 osu! 里打开（自动导入）。

不需要显卡。3 分钟的歌、一个难度，在现代 CPU 上约 1 分钟；16 核 CPU 实测 70 秒，RTX 4090 上 16 秒。
Windows 10 / 11，64 位。

### Python 安装

```bash
git clone https://github.com/kanze1/AUTO-OSU
cd AUTO-OSU
python -m venv .venv && .venv\Scripts\activate       # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -e .[gui]
# 有 NVIDIA 显卡（可选）：pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m autoosu --download                          # 第一次下载模型（约 320 MB）
python -m autoosu                                     # 打开图形界面
python -m autoosu "歌曲.mp3" -d Normal Hard -o out    # 命令行
```

Python 3.10 及以上。macOS / Linux 可以用 Python 方式运行，exe 只提供 Windows 版。

## 界面说明

| 区域 | 说明 |
| --- | --- |
| 歌曲 | 拖放或浏览选择音频文件。非 mp3 / ogg 的格式会自动转成 mp3 打进 `.osz`（自带 ffmpeg）。 |
| 难度 | Easy / Normal / Hard / Insane，可多选，全部打进同一个 `.osz`。默认 Normal + Hard。 |
| 输出 | 保存目录；「生成后自动导入 osu!」会直接打开 `.osz`，等于双击它。 |
| 模型 | 显示模型是否就绪；缺失时一键下载（自动校验 sha256）。 |
| 右上角 | 中 / 英切换，亮 / 暗切换。设置和上次用的歌、目录都会记住。 |

**高级选项**（点「高级选项」展开）：

| 选项 | 说明 |
| --- | --- |
| 随机种子 | 同一首歌换个数字得到另一版摆放；同一个种子结果可复现。 |
| BPM / 偏移 | 留空自动检测；检测错了（常见于变速歌或前奏很空的歌）手动填。偏移单位 ms，填的是分析层的值，写入 `.osu` 时会再按 osu! 惯例提前 26 ms。 |
| 谱师名 | 写进 `.osu` 的 Creator，默认 AUTO-OSU。 |
| 星级条件 | 给两个模型的星级提示，留空按难度默认：Easy 2.0 / Normal 3.2 / Hard 4.5 / Insane 5.5。想让 Hard 更密就填大一点。 |
| 摆放质量 | 坐标模型的扩散步数：快速 50 / 标准 100 / 精细 200。步数越多越慢，标准档已经够用。 |
| 计算设备 | auto 会优先用 CUDA 显卡；没有就用 CPU。 |
| 生成引擎 | 「AI 模型」是正常模式；「纯规则」不用模型、几秒出图，只在没模型或想对比时用。 |
| 试听 mp3 | 另存一个原曲压低音量、每个物件加点击声的 mp3，不开 osu! 也能听节奏对不对。 |
| 顶部波纹动画 | 待机时几乎不占资源（见下文）；机器很弱可以关掉。 |

## 命令行

`python -m autoosu 歌曲 [选项]`（exe 也支持同样的参数：`AUTO-OSU.exe 歌曲.mp3 -d Hard`）。

| 选项 | 说明 |
| --- | --- |
| `-d Easy Normal Hard Insane` | 要生成的难度 |
| `-o 目录` | 输出目录，默认 `out` |
| `--seed N` | 随机种子 |
| `--bpm` / `--offset` | 手动 BPM / 红线偏移（ms） |
| `--title` / `--artist` / `--creator` | 覆盖元数据（默认从音频标签或「歌手 - 歌名」文件名读取） |
| `--star X` | 星级条件 |
| `--coord-steps N` | 扩散步数，默认 100 |
| `--cfg-scale X` | 坐标模型的 classifier-free guidance，默认 1.0 |
| `--temperature` / `--density` / `--density-bias` / `--decode-steps` | 节奏模型采样参数：温度、目标每小节物件数、"不放"偏置（负数更密）、解码轮数 |
| `--device auto\|cuda\|cpu` | 计算设备 |
| `--rules` | 纯规则模式 |
| `--rhythm-model` / `--coord-model` | 指定模型文件；不指定则在 `models/` 里找 |
| `--no-coord-model` | 只用节奏模型，摆放走规则 |
| `--download` | 缺模型时从 GitHub Release 下载 |
| `--osu-shift 26` | 写入 `.osu` 时物件提前多少 ms |
| `--preview` | 另存试听 mp3 |
| `--debug-plot` | 另存一张分析图：响度与 kiai 段、onset 与拍线、各难度选中的音符 |
| `--dump-events` | 打印每个物件的时间、类型、拍位 |

## 原理

```
歌曲 ──► 音频分析 ──► 定时 ──► 节奏模型 ──► 坐标模型 ──► 滑条贴合 ──► .osz
       librosa：HPSS、  BPM、偏移、  什么时候按       放在哪里      每条滑条都在
       onset、响度      kiai 段      （tick 变换器）  （扩散 DiT）    屏幕内
```

**音频分析与定时（规则）。** 把打击和旋律分离，分频段（底鼓 / 军鼓 / 镲）算 onset 包络；tempogram 估 BPM 并纠正倍频；
在波形上做 1 ms 精度的偏移校准；用底鼓、和声变化和响度找小节起拍；按响度分段找 kiai。
物件统一比音频瞬态提前 26 ms 写入，这是 ranked 谱面的普遍惯例（玩家的偏移设置都是按它校准的）。

**节奏模型** `rhythm_v0.pt`，约 2900 万参数。在 1/4 拍网格上的双向 Transformer：每个 tick 看 ±80 ms 的梅尔频谱、
它在小节里的位置、局部响度，加上要求的星级 / CS / AR / OD / HP，预测六类之一（无 / 圈 / 滑条头 / 滑条身 / 滑条尾 / 转盘），
MaskGIT 式 12 轮并行解码。训练语料是 HuggingFace `project-riz/osu-beatmaps` 里的 139,582 张 osu!standard 谱面。

**坐标模型** `coord_v0.pt`，约 1.3 亿参数。结构是 [osu-diffusion](https://github.com/OliBomby/osu-diffusion) 的 DiT-B，
本项目用完整的 1000 步噪声计划从零训练了 20 万步。输入是物件 token 序列（圈、滑条头、锚点、滑条尾、转盘，各带时间），
从纯噪声去噪出每个点的 x / y；条件只有星级和 CS。训练时一半样本把「到上一个点的距离」置零，所以它会自己决定跨度。

**滑条贴合。** 模型画的是滑条的形状，长度由节奏决定。每条滑条围绕滑条头缩放到要求的像素长度；如果路径会出场地，
先试镜像弯向、再绕头旋转（小角度优先），实在放不下才缩短并加一条本地绿线保持时长不变。快歌会按 BPM 压低 SliderMultiplier。

设计记录和实验数据在 [docs/rhythm_model_design.md](docs/rhythm_model_design.md)，
Mapperatorinator 的技术分析在 [docs/mapperatorinator_tech.md](docs/mapperatorinator_tech.md)。

### 顶部动画是怎么做的

金色脉冲波纹是 PIL 以 2 倍纵向分辨率画线再缩小（抗锯齿），由后台线程按显示器刷新率发节拍
（240 Hz 可跑满；Tk 自带的 `after()` 在这种频率下会空转吃掉四分之一个核心）。待机时播放预渲染的无缝循环，
唯一帧率按 64 MB 内存预算自适应（1440p、150% 缩放下约 96 帧/秒），实测占约 7% 单核；生成时改为实时渲染，
让波峰跟着进度走，每帧约 1.4 ms。

## 自己训练

训练发布模型用到的全部代码都在仓库里，流程在 [docs/rhythm_model_design.md](docs/rhythm_model_design.md)：

| 文件 | 作用 |
| --- | --- |
| `autoosu/ml/prepare_data.py` | HuggingFace 分片 → log-mel 特征 + tick 标签 |
| `autoosu/ml/train.py` | 节奏模型训练（`torchrun` 多卡 DDP，`--mode masked`，wandb 记录） |
| `coord/` + `scripts/coord_make_ors.py` | 坐标模型训练（accelerate；配置 `coord/configs/diffusion/coord_v0.yaml`） |
| `scripts/server_*.sh` | 服务器上的环境、数据准备、启动训练 |
| `scripts/coord_export.py` | 检查点 → 发布用模型文件 |

两个模型文件都能用 `torch.load(weights_only=True)` 加载，里面没有任何 pickle 代码。

## 打包 exe

```powershell
pip install -e .[build]
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # dist/AUTO-OSU-<版本>-win64-cpu.zip
```

用装了 CUDA 版 torch 的环境加 `-Venv .venv-gpu -Suffix cuda` 可以打显卡版。
头像 / 图标：`python scripts/make_icon.py 头像.png` 会生成窗口头像、窗口图标和 exe 图标。

## 常见问题

**模型下载失败？** 手动从 [models-v0](https://github.com/kanze1/AUTO-OSU/releases/tag/models-v0) 下载 `rhythm_v0.pt` 和 `coord_v0.pt`，
放到 exe 旁边的 `models/` 文件夹（或 `~/.autoosu/models/`）。

**杀毒软件报毒？** PyInstaller 打的包常被误报。可以用 Python 方式运行，或自己按上面的步骤打包。

**BPM 或偏移不对？** 在高级选项里手动填。变速歌目前只会给一条红线。

**osu! 没有自动打开？** 说明系统里 `.osz` 没有关联到 osu!，把生成的 `.osz` 拖进 osu! 窗口即可。

**生成很慢？** CPU 模式一个难度约一分钟属于正常；摆放质量选「快速」能快一倍；有 NVIDIA 显卡的话用 Python 方式装 CUDA 版 torch。

**动画卡或者想省资源？** 高级选项里关掉「顶部波纹动画」。

## 路线图

- 坐标模型 v1：把滑条要求长度作为逐点条件，根治长滑条缩短。
- 节奏模型：谱师风格条件、1/12 网格覆盖三连音。
- 同一首歌多种子出多版供挑选；显卡版 zip。

## 致谢

- [osu-diffusion](https://github.com/OliBomby/osu-diffusion)（MIT）—— DiT 结构与扩散代码，收录在 `autoosu/ml/coord`。
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)（MIT）—— token 化思路和对比基线。
- [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) —— 训练语料。
- [osu-dreamer](https://github.com/jaswon/osu-dreamer) —— 早期基线。

## 许可

MIT，见 [LICENSE](LICENSE)。生成的谱面归你，歌曲归原作者。作者：kanzei。
