"""Tiny bilingual string table for the GUI (zh-CN / en)."""
from __future__ import annotations

from typing import Dict

LANGS = ("zh", "en")
_current = "zh"

STRINGS: Dict[str, Dict[str, str]] = {
    "app.title": {"zh": "AUTO-OSU 自动谱面生成", "en": "AUTO-OSU beatmap generator"},
    "app.subtitle": {"zh": "选一首歌，几十秒后拿到可以直接玩的 osu! 谱面",
                     "en": "Pick a song, get a playable osu! beatmap in under a minute"},
    "song.section": {"zh": "歌曲", "en": "Song"},
    "song.hint": {"zh": "把音频文件拖到这里，或点击「浏览」（支持 mp3 / ogg / wav / flac / m4a）",
                  "en": "Drop an audio file here or click Browse (mp3 / ogg / wav / flac / m4a)"},
    "song.browse": {"zh": "浏览…", "en": "Browse…"},
    "song.filetypes": {"zh": "音频文件", "en": "Audio files"},
    "diff.section": {"zh": "难度", "en": "Difficulties"},
    "diff.Easy": {"zh": "Easy　新手", "en": "Easy"},
    "diff.Normal": {"zh": "Normal　普通", "en": "Normal"},
    "diff.Hard": {"zh": "Hard　困难", "en": "Hard"},
    "diff.Insane": {"zh": "Insane　疯狂", "en": "Insane"},
    "diff.hint": {"zh": "一次可以生成多个难度，都会打包进同一个 .osz", "en": "Several difficulties go into one .osz"},
    "out.section": {"zh": "输出", "en": "Output"},
    "out.folder": {"zh": "保存到", "en": "Save to"},
    "out.browse": {"zh": "选择文件夹…", "en": "Choose folder…"},
    "out.open_osu": {"zh": "生成后自动导入 osu!（打开 .osz）", "en": "Import into osu! when done (opens the .osz)"},
    "adv.section": {"zh": "高级选项", "en": "Advanced"},
    "adv.show": {"zh": "▸ 高级选项", "en": "▸ Advanced options"},
    "adv.hide": {"zh": "▾ 高级选项", "en": "▾ Advanced options"},
    "adv.seed": {"zh": "随机种子", "en": "Seed"},
    "adv.seed_hint": {"zh": "换个数字得到另一版摆放", "en": "change it for a different layout"},
    "adv.bpm": {"zh": "BPM（留空自动检测）", "en": "BPM (blank = detect)"},
    "adv.offset": {"zh": "偏移 ms（留空自动检测）", "en": "Offset ms (blank = detect)"},
    "adv.creator": {"zh": "谱师名（Creator）", "en": "Creator name"},
    "adv.quality": {"zh": "摆放质量", "en": "Placement quality"},
    "adv.quality.fast": {"zh": "快速（50 步）", "en": "Fast (50 steps)"},
    "adv.quality.normal": {"zh": "标准（100 步）", "en": "Standard (100 steps)"},
    "adv.quality.high": {"zh": "精细（200 步）", "en": "Fine (200 steps)"},
    "adv.device": {"zh": "计算设备", "en": "Device"},
    "adv.device.auto": {"zh": "自动", "en": "auto"},
    "adv.engine": {"zh": "生成引擎", "en": "Engine"},
    "adv.engine.ml": {"zh": "AI 模型（推荐）", "en": "AI models (recommended)"},
    "adv.engine.rules": {"zh": "纯规则（无需模型，效果一般）", "en": "Rules only (no models, basic)"},
    "adv.star": {"zh": "星级条件（留空按难度默认）", "en": "Star rating (blank = per difficulty)"},
    "adv.preview": {"zh": "另存带点击声的试听 mp3", "en": "Also save a click-track preview mp3"},
    "models.section": {"zh": "模型", "en": "Models"},
    "models.ok": {"zh": "模型已就绪：节奏 {rhythm} · 坐标 {coord}", "en": "Models ready: rhythm {rhythm} · coordinates {coord}"},
    "models.missing": {"zh": "缺少模型文件（约 320 MB），点击下载后才能使用 AI 模式",
                       "en": "Model files missing (~320 MB); download them to use the AI engine"},
    "models.download": {"zh": "下载模型", "en": "Download models"},
    "models.downloading": {"zh": "正在下载 {name}…", "en": "Downloading {name}…"},
    "models.folder": {"zh": "模型文件夹", "en": "Models folder"},
    "run.generate": {"zh": "生成谱面", "en": "Generate"},
    "run.running": {"zh": "生成中…", "en": "Generating…"},
    "run.open_folder": {"zh": "打开输出文件夹", "en": "Open output folder"},
    "run.open_osz": {"zh": "在 osu! 中打开", "en": "Open in osu!"},
    "status.idle": {"zh": "就绪", "en": "Ready"},
    "status.load": {"zh": "读取音频…", "en": "Loading audio…"},
    "status.analyse": {"zh": "分析音频（鼓点、旋律、段落）…", "en": "Analysing audio (drums, melody, sections)…"},
    "status.timing": {"zh": "估计 BPM 和偏移…", "en": "Estimating BPM and offset…"},
    "status.load rhythm model": {"zh": "加载节奏模型…", "en": "Loading rhythm model…"},
    "status.load coordinate model": {"zh": "加载坐标模型…", "en": "Loading coordinate model…"},
    "status.rhythm": {"zh": "{diff}：生成节奏…", "en": "{diff}: generating rhythm…"},
    "status.placing": {"zh": "{diff}：摆放物件 {detail}", "en": "{diff}: placing objects {detail}"},
    "status.package": {"zh": "打包 .osz…", "en": "Packing the .osz…"},
    "status.done": {"zh": "完成：{file}（{secs:.0f} 秒，{device}）", "en": "Done: {file} ({secs:.0f} s, {device})"},
    "status.error": {"zh": "出错了：{err}", "en": "Error: {err}"},
    "result.summary": {"zh": "BPM {bpm:g}，{n} 个难度：", "en": "BPM {bpm:g}, {n} difficulties:"},
    "result.diff": {"zh": "  {name}：{objects} 个物件（{sliders} 滑条），{nps:.1f} 个/秒",
                    "en": "  {name}: {objects} objects ({sliders} sliders), {nps:.1f}/s"},
    "err.no_song": {"zh": "请先选择一首歌", "en": "Choose a song first"},
    "err.no_diff": {"zh": "至少勾选一个难度", "en": "Tick at least one difficulty"},
    "err.no_models": {"zh": "AI 模式需要先下载模型，或在高级选项里切换到纯规则模式",
                      "en": "The AI engine needs the models; download them or switch to rules-only in Advanced"},
    "err.file_missing": {"zh": "文件不存在：{path}", "en": "File not found: {path}"},
    "lang.toggle": {"zh": "English", "en": "中文"},
    "about": {"zh": "开源项目 · github.com/kanze1/AUTO-OSU", "en": "Open source · github.com/kanze1/AUTO-OSU"},
}


def set_language(lang: str) -> None:
    global _current
    _current = lang if lang in LANGS else "en"


def language() -> str:
    return _current


def tr(key: str, **kwargs) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(_current) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text
