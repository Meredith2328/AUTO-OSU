# Mapperatorinator 技术栈笔记（v32，2026-09 读源码整理）

代码：`external/Mapperatorinator`，MIT 许可。作者 OliBomby，基于 osuT5（gyataro）和 osu-diffusion。

## 1. 训练数据

- HuggingFace 数据集 `project-riz/osu-beatmaps`（WebDataset 格式，公开、不需要登录）：
  ranked/loved 谱面 **213,068 张**，独立音轨 46,386 首，2007-10 到 2025-12，谱面总时长 8,419 小时。
  `compressed` 变体（64 kbps 单声道 opus）79 GB / 74 个 tar；`original` 变体 220 GB。
  每个样本 = 一首歌的音频 + 一个 json（该歌所有 .osu 原文 + 元数据：beatmap_id、approved、星级等）。
- v32 用 0..72 号分片训练、72..74 做验证（`configs/train/v32.yaml`）。
- 附加标签：`project-riz/osu-beatmap-tags`（谱面的用户标签，如 jump aim / stream / tech，出现 ≥ 2 次才保留）、
  谱师 ID 表（`datasets/beatmap_users.json`）。这些成为条件 token。
- 数据增强：DT 变速 1.0-1.2x（概率 0.3）、随机帧偏移（0.5）、水平/垂直镜像（各 0.5）、随机吸附噪声（0.25）、谱师条件 dropout 0.1。

## 2. 输入表示

音频 → 16 kHz 单声道（归一化）→ log-mel 频谱 128 维，hop 128 采样 = **8 ms 一帧**。
一个窗口 2048 帧 = **16.4 s**。整首歌切成滑动窗口，推理时前一窗的输出 token 作为上下文（lookback 0.5），
每窗只生成前 60%（lookahead 0.4 只看不生成）。

## 3. 输出表示：把谱面写成一门"语言"

`osuT5/osuT5/event.py` + `tokenizer.py`，词表 4069 个 token。每个物件是一组 token：

| 类别 | token | 说明 |
| --- | --- | --- |
| 时间 | `TIME_SHIFT` 0..1638 | **10 ms 分辨率**，窗口内绝对时间 |
| 吸附 | `SNAPPING` 0..16 | 这个物件吸附在哪个拍子分母上（1/1、1/2、1/3、1/4…），后处理靠它重新吸附 |
| 类型 | `CIRCLE` / `SLIDER_HEAD` / `BEZIER_ANCHOR` / `PERFECT_ANCHOR` / `CATMULL_ANCHOR` / `RED_ANCHOR` / `LAST_ANCHOR` / `SLIDER_END` / `SPINNER` / `SPINNER_END` | 滑条按锚点逐个生成 |
| 位置 | `POS` 0..191 (+ `POS_REFINE`) | 32 px 网格 16×12 格，再来一轮细化 token 提高精度 |
| 其它 | `NEW_COMBO`、`HITSOUND`(72 种组合)、`VOLUME`、SV、`SLIDER_SUSTAIN` 等 | 长物件每 8 s 插一个 sustain token |
| 节拍 | `BEAT` / `MEASURE` / `TIMING_POINT` | timing 作为单独的输出类型先生成 |

条件 token（前缀）：游戏模式、**难度星级**（分箱到 24-26 级，上限 12 星）、谱师 ID、年份（2007-2024）、
是否有 hitsound、歌曲长度、歌曲位置、全局 SV、CS、标签描述符、输出上下文类型。
v32 的输出顺序：`[TIMING, MAP, SV]`，即先出红线，再出物件，最后出绿线。

`add_distances: false`、`add_positions: true`：位置直接由 token 决定；
osu-diffusion（DiT-B 扩散模型，只负责坐标）在 v32 推理配置里 `generate_positions: false`，默认不用。

## 4. 模型

`modeling_mapperatorinator.py`：Whisper 风格的 encoder-decoder（HF `WhisperForConditionalGeneration` 改造，
名字 `varwhisper-small`）。d_model 768，编码器 12 层 + 解码器 12 层，12 头，FFN 3072，
局部注意力窗口 128 + 每层全局注意力，RoPE 位置编码。
**2.16 亿参数**（编码器 8700 万，解码器 1.26 亿），bf16，flash-attention 2。
每个游戏模式一份权重（`gamemode=0` 是 std，0.87 GB），另有一份基础模型专门生成 timing。

训练：Muon 优化器，lr 0.002，batch 32 × 梯度累积 2，**70 万步**，label smoothing 0.2。
作者称总计约 5700 GPU 小时（4060 Ti + 租的 4090）。

## 5. 推理流水线（`inference.py` → `Processor` → `Postprocessor`）

1. 编码器一次性算完所有窗口（151 个窗口 1.8 s）。
2. 先生成 timing（低温度 0.1），再逐窗自回归生成物件：temperature 0.9、top_p 0.9，
   带 CFG（`cfg_scale > 1` 时对谱师/描述符做 classifier-free guidance，需要额外一次无条件前向）。
3. Logit 处理器：`TimeshiftBias`（给所有 TIME_SHIFT token 加一个常数偏置）、
   `MonotonicTimeShift`（时间只能向前）、`LookbackBias`（修正上下文窗口带来的频率偏差）、条件温度。
4. CUDA-graph 快速解码循环（`fast_decoder_loop`），4 分钟歌全流程约 1-2 分钟。
5. 后处理：按 `SNAPPING` token 把时间**重新吸附到红线网格**（`resnap_events`）；
   滑条锚点 → `SliderPath` 计算长度 → `get_human_sv_and_length` 把 SV/长度调成人类谱师会用的数值；
   可选把几乎重叠的坐标吸成完全重叠（v32 关闭）；导出 .osu / .osz。

## 6. 密度由谁决定，以及可以动的旋钮

密度 = 解码器在每一步选择"发 TIME_SHIFT 跳过时间"还是"发一个物件"，这是从 ranked 谱面学来的分布，
受条件 token 影响。我们这首歌 4.0 星 → 2.6 物件/秒，5.5 星 → 3.2，确实偏稀。可试：

| 旋钮 | 作用 | 备注 |
| --- | --- | --- |
| `difficulty` | 星级越高越密 | 试 6.5-7.5 |
| `descriptors` | 风格标签，如 `stream`、`complex`、`speed`、`spaced` | 配合 `cfg_scale=2~3`，`negative_descriptors` 放 `simple` |
| `timeshift_bias` | 负值 = 不爱跳过时间 = 更密 | 试 -0.5 到 -2，太负会乱 |
| `mapper_id` | 模仿某位谱师 | 需要 osu! 用户 ID，配合 cfg_scale |
| `year` | 现代谱面更密、更 jump | 2023-2024 |
| `temperature` / `top_p` | 随机性 | 降温更保守 |
| LoRA 微调 | `configs/train/lora_v32.yaml`，用自己挑的谱面训风格 | 4090 可跑，这是"改进他们方案"最直接的路 |

## 7. 相关文件索引

- `configs/inference/default.yaml`：全部推理参数及解释
- `configs/train/v32.yaml` + `configs/model/varwhisper_small_v3.yaml`：训练与模型配置
- `osuT5/osuT5/tokenizer.py`：token 定义（时间 10 ms、位置 32 px、难度分箱）
- `osuT5/osuT5/inference/processor.py`：窗口滑动、采样、上下文
- `osuT5/osuT5/inference/postprocessor.py`：重吸附、滑条长度、导出
- `osuT5/osuT5/dataset/web_dataset.py`：HF 数据集的读取与 token 化
- `osu_diffusion/`：可选的坐标扩散模型
