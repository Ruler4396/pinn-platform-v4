"""绘图相关通用工具。"""

from __future__ import annotations

from pathlib import Path


中文字体候选文件 = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)


def 获取中文字体文件() -> Path:
    for path in 中文字体候选文件:
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到可用中文字体文件；请确认系统已安装 Noto Sans/Serif CJK 或 WenQuanYi Zen Hei。"
    )


def 配置中文绘图(prefer_serif: bool = False) -> tuple[Path, str]:
    """配置 Matplotlib 使用显式中文字体文件，避免中文乱码。"""

    import matplotlib as mpl  # type: ignore
    from matplotlib import font_manager  # type: ignore

    font_path = 获取中文字体文件()
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()

    if prefer_serif:
        mpl.rcParams["font.family"] = [font_name]
        mpl.rcParams["font.serif"] = [font_name, "DejaVu Serif"]
    else:
        mpl.rcParams["font.family"] = [font_name]
        mpl.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["svg.fonttype"] = "none"
    return font_path, font_name
