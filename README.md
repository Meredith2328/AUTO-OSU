# AUTO-OSU · osu!mania 4K

[English](README.en.md) · [设计与评估](docs/mania4k_engine.md) · [原版 README（osu!standard）](docs/original/README.md)

**丢进一首歌，得到一张卡点准确、按星级校准的 osu!mania 4K 谱面。**

本仓库 fork 自 [kanze1/AUTO-OSU](https://github.com/kanze1/AUTO-OSU)。原项目生成 **osu!standard** 谱面（节奏 Transformer + 坐标扩散模型、桌面 GUI、Windows exe），这些功能全部保留，说明见[原版 README](docs/original/README.md)。本 fork 的工作集中在 **osu!mania 4K**。

## 使用

```bash
pip install -e .                       # Python 3.10+；CPU 即可
python -m autoosu song.mp3 --mode mania4k -d Easy Normal Hard Insane Expert -o out
python -m autoosu song.mp3 --mode mania4k -d Hard --mania-stars 3.5     # 指定星级
python -m autoosu                       # GUI，游戏模式选 “osu!mania 4K”
```

输出 `.osz`，双击导入 osu!/lazer。首次运行会下载约 80 MB 的节拍追踪权重（也可把 `beat_this-final0.ckpt` 放进 `models/`）。一首 3 分钟的歌、5 个难度约 1–2 分钟（单线程 CPU）。

## 它做了什么

| 环节 | 做法 |
| --- | --- |
| 定时 | Beat This! 神经节拍追踪 + 让全曲起音对齐 1/4 网格的 BPM/相位精搜；变速检测、多红线、BPM 取整、红线放重拍；按排位谱实测的 24 ms 约定写入 |
| 卡点（硬约束） | 每个音符都在网格上，且 ±8 ms 内有明显起音；同一拍内不混用直/三连节奏；写出前 `verify_chart` 逐谱复查，不合格不输出 |
| 音符 | 在节拍网格上的 TCN（按星级条件化），从 300 组排位 4K 谱学习放哪、几押、是否长按 |
| 配置 | 从 150 万行排位谱学习的轨道配置模型，在长按占轨、同轨最短间隔等硬约束下解码 |
| 难度 | 用 osu! 同款星级算法（rosu-pp）二分校准；OD/HP、双押比例取同星级排位谱统计；撑不起的难度会跳过而不是硬塞 |

## 质量（v1，未见过的 22 首歌 / 67 张排位谱对照）

| | 旧规则引擎 | v1 | 排位谱 |
| --- | --- | --- | --- |
| 音符落在人工谱网格上 | 95.8 % | **98.4 %** | |
| 通过同步/可玩性检查 | 61 % | **100 %** | |
| 星级误差 | 1.44★ | **0.21★** | |
| 每秒音符 / 双押行 | 3.5 / 9.9 % | **8.6 / 35.1 %** | 9.2 / 35.6 % |
| 定时与人工红线一致（218 首单 BPM 歌） | 77.9 % | **93.5 %** | |

## 迭代

- **v1**：精确定时 + 卡点硬约束 + 星级校准。人工评分 70/100：卡点与难度到位，但键型单一（几乎都是切）、段落感弱、感受不到情绪起伏。
- **v2（进行中）**：从排位谱学习段落、情绪强度与键型（叠/切/乱及细分）的关系，按段落规划键型与强度。

已知限制：持续漂移速度的现场录音、swing/爵士、多次变速的比赛曲定时较弱；不做 SV 与键音。

## 复现

语料抓取、训练与评估脚本在 `scripts/mania4k_*.py`，完整方法与数据见 [docs/mania4k_engine.md](docs/mania4k_engine.md)。

许可证：MIT + 署名条款，见 [LICENSE](LICENSE)。
