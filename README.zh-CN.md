# AUTO-OSU

**把任意一首歌变成能直接玩的 osu!standard 谱面。** 拖进一个 mp3，勾选想要的难度，点「生成谱面」，
一分钟后 `.osz` 自动导入 osu!。

[English](README.md) · [下载](https://github.com/kanze1/AUTO-OSU/releases) · [原理](#原理)

![AUTO-OSU 界面](docs/screenshot_zh.png)

<details><summary>亮色主题 / English，以及生成中的顶部脉冲</summary>

![亮色主题](docs/screenshot_en.png)

![生成中](docs/screenshot_busy.png)

</details>

## 下载即用（Windows，免安装）

1. 到 [Releases](https://github.com/kanze1/AUTO-OSU/releases) 下载 `AUTO-OSU-<版本>-win64-cpu.zip`，解压到任意位置。
2. 双击 `AUTO-OSU.exe`。
3. 把歌曲拖进窗口（mp3 / ogg / wav / flac / m4a 都行），勾选 **Normal** / **Hard** 等难度，点 **生成谱面**。
4. `.osz` 会写到 exe 旁边的 `output` 文件夹，并默认直接在 osu! 里打开（自动导入）。

压缩包里已经带了两个训练好的模型（约 320 MB）；如果缺失，窗口里有一键下载。
不需要显卡：一首 3 分钟的歌在现代 CPU 上约 1 分钟，NVIDIA 显卡上约 15 秒（用 `-cuda` 包或 Python 安装方式）。

界面中英双语、亮色/暗色两套（右上角按钮切换）；默认跟随系统语言和主题。

## Python 安装

```bash
git clone https://github.com/kanze1/AUTO-OSU
cd AUTO-OSU
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -e .[gui]
# 显卡（可选）：pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m autoosu --download                          # 第一次下载模型（约 320 MB）
python -m autoosu                                     # 打开图形界面
python -m autoosu "歌曲.mp3" -d Normal Hard -o out    # 命令行
```

常用命令行参数：

| 参数 | 说明 |
| --- | --- |
| `-d Easy Normal Hard Insane` | 要生成的难度，多个难度打进同一个 `.osz` |
| `--seed N` | 换一个数字得到另一版摆放 |
| `--bpm` / `--offset` | 手动指定 BPM / 红线偏移（ms），检测不准时用 |
| `--star X` | 给模型的星级条件（默认按难度：2.0 / 3.2 / 4.5 / 5.5） |
| `--coord-steps 50` | 减少扩散步数，摆放更快（默认 100） |
| `--device cpu` | 强制用 CPU |
| `--rules` | 只用规则版（不需要模型，快但效果一般） |
| `--preview` | 另存一个带点击声的 mp3，不开 osu! 也能听节奏对不对 |

## 原理

```
歌曲 ──► 音频分析 ──► 定时 ──► 节奏模型 ──► 坐标模型 ──► 滑条贴合 ──► .osz
        （librosa：HPSS、  BPM、偏移、   什么时候按     按在哪里       保证每条滑条
          onset、响度）     kiai 段      （tick 变换器） （扩散 DiT）    都在屏幕内
```

* **音频分析 + 定时**（规则）：打击/旋律分离，分频段 onset 包络，tempogram 估 BPM 并纠正倍频，
  波形级 1 ms 偏移校准，小节起拍相位，响度分段 → kiai。物件统一提前 26 ms 写入，这是 osu! ranked
  谱面的惯例。
* **节奏模型** —— `rhythm_v0.pt`，2900 万参数。1/4 拍网格上的双向 Transformer，每个 tick 根据
  ±80 ms 的 mel 频谱、节拍位置、响度和要求的星级 / CS / AR / OD / HP 预测类别
  （无 / 圈 / 滑条头 / 滑条身 / 滑条尾 / 转盘），MaskGIT 式 12 轮并行解码。
  在约 14 万张 ranked osu!standard 谱面上训练；与人类谱面的 onset F1 为 0.96
  （同一首歌两个人类难度之间约 0.74）。
* **坐标模型** —— `coord_v0.pt`，1.3 亿参数（DiT-B，结构来自
  [osu-diffusion](https://github.com/OliBomby/osu-diffusion)，本项目在完整 1000 步噪声计划上从零训练）。
  输入物件 token 序列（圈、滑条头、锚点、滑条尾、转盘及各自时间），从纯噪声去噪出 x/y 坐标，
  条件是星级和 CS。默认 100 步采样。
* **滑条贴合**：模型画的是滑条的*形状*，长度由节奏决定。每条滑条围绕头部缩放到要求长度；
  若路径会出场，先镜像 / 旋转形状，实在放不下才缩短并加一条本地绿线保证时长不变。
  每张生成的谱面都会检查物件是否在屏幕内。

设计记录、实验和结果：[docs/rhythm_model_design.md](docs/rhythm_model_design.md)、
[docs/mapperatorinator_tech.md](docs/mapperatorinator_tech.md)。

## 自己训练

训练发布模型用到的所有东西都在仓库里：

* `autoosu/ml/prepare_data.py` —— HuggingFace `project-riz/osu-beatmaps` 分片 → log-mel + tick 标签
* `autoosu/ml/train.py` —— 节奏模型（`torchrun` DDP，`--mode masked`，wandb 记录）
* `coord/` + `scripts/coord_make_ors.py` —— 坐标模型（accelerate，配置 `coord/configs/diffusion/coord_v0.yaml`）
* `scripts/server_*.sh` —— 服务器上的完整流程（环境、数据、训练）
* `scripts/coord_export.py` / `autoosu.ml.coord_infer.export_coord_model` —— 检查点 → 发布文件

两个模型都用 `torch.load(weights_only=True)` 加载，不含任何 pickle 代码。

## 打包 exe

```powershell
pip install -e .[build]
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # dist/AUTO-OSU-<版本>-win64-cpu.zip
```

用装了 CUDA 版 torch 的环境加 `-Venv .venv-gpu -Suffix cuda` 可打显卡版。

## 已知限制（v0）

* 一首歌只有一条红线：变速歌会被当成单一 BPM，特殊情况用 `--bpm/--offset`。
* 滑条*形状*是学出来的，但滑条*长度*还不是模型输入，快歌上的长滑条偶尔会被缩短（绿线保证时长正确）。
* 打击音效只有基于鼓的简单 whistle / clap / finish；没有背景图和 storyboard。
* 定时是自动检测的，不是人工校对的：要投稿之前请在编辑器里核对偏移。

## 致谢

* [osu-diffusion](https://github.com/OliBomby/osu-diffusion)（MIT）—— DiT 结构与扩散代码，收录在 `autoosu/ml/coord`。
* [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)（MIT）—— token 化思路和对比基线。
* [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) —— 训练语料。
* [osu-dreamer](https://github.com/jaswon/osu-dreamer) —— 早期基线。

## 许可

MIT，见 [LICENSE](LICENSE)。生成的谱面归你，歌曲归原作者。
