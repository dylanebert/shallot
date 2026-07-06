#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen


@dataclass
class Color:
    fill: str
    shadow: str
    highlight: str
    deep_shadow: str = "#B87654"
    outline: str = "#6B4230"
    # wordmark fill — matches the site's fg text token, not the warm icon
    text: str = "#F0ECE8"


DEFAULT_COLOR = Color(
    fill="#E8A86B",
    shadow="#C4784A",
    highlight="#F5D4B8",
    deep_shadow="#B87654",
    outline="#6B4230",
    text="#F0ECE8",
)


@dataclass
class IconPaths:
    main: str
    clove_left: str
    clove_right: str
    center_crease: str
    shade: str
    highlight: str


def generate_icon_paths() -> IconPaths:
    cx, cy = 40, 40
    top_y, bottom_y, mid_y = 2, 78, 46
    top_width, max_width = 4, 26

    top = (cx, top_y)
    right_mid = (cx + max_width, mid_y)
    bottom = (cx, bottom_y)
    left_mid = (cx - max_width, mid_y)

    top_right_c1 = (cx + top_width, top_y + 8)
    top_right_c2 = (cx + max_width, mid_y - 18)
    right_bottom_c1 = (cx + max_width, mid_y + 14)
    right_bottom_c2 = (cx + 8, bottom_y - 8)
    bottom_left_c1 = (cx - 8, bottom_y - 8)
    bottom_left_c2 = (cx - max_width, mid_y + 14)
    left_top_c1 = (cx - max_width, mid_y - 18)
    left_top_c2 = (cx - top_width, top_y + 8)

    main = (
        f"M{top[0]},{top[1]} "
        f"C{top_right_c1[0]},{top_right_c1[1]} {top_right_c2[0]},{top_right_c2[1]} {right_mid[0]},{right_mid[1]} "
        f"C{right_bottom_c1[0]},{right_bottom_c1[1]} {right_bottom_c2[0]},{right_bottom_c2[1]} {bottom[0]},{bottom[1]} "
        f"C{bottom_left_c1[0]},{bottom_left_c1[1]} {bottom_left_c2[0]},{bottom_left_c2[1]} {left_mid[0]},{left_mid[1]} "
        f"C{left_top_c1[0]},{left_top_c1[1]} {left_top_c2[0]},{left_top_c2[1]} {top[0]},{top[1]} Z"
    )

    shade_inset = 5
    shade = (
        f"M{bottom[0]},{bottom[1]} "
        f"C{right_bottom_c2[0]},{right_bottom_c2[1]} {right_bottom_c1[0]},{right_bottom_c1[1]} {right_mid[0]},{right_mid[1]} "
        f"C{right_mid[0]-shade_inset},{mid_y+12} {cx+4},{bottom_y-8} {cx},{bottom_y-shade_inset} Z"
    )

    highlight_inset = 5
    highlight = (
        f"M{top[0]},{top[1]} "
        f"C{left_top_c2[0]},{left_top_c2[1]} {left_top_c1[0]},{left_top_c1[1]} {left_mid[0]},{left_mid[1]} "
        f"C{left_mid[0]+highlight_inset},{mid_y-16} {cx-top_width+highlight_inset},{top_y+6} {cx},{top_y+highlight_inset} Z"
    )

    center_crease = f"M{cx},{top_y + 6} C{cx},{top_y + 18} {cx},{bottom_y - 28} {cx},{bottom_y - 6}"

    clove_left = (
        f"M{cx},{top_y + 4} "
        f"C{cx - 3},{top_y + 12} {cx - 18},{mid_y - 18} {cx - 20},{mid_y - 2} "
        f"C{cx - 20},{mid_y + 6} {cx - 12},{bottom_y - 16} {cx - 4},{bottom_y - 8} "
        f"C{cx - 6},{bottom_y - 20} {cx - 14},{mid_y} {cx - 14},{mid_y - 8} "
        f"C{cx - 14},{mid_y - 20} {cx - 2},{top_y + 10} {cx},{top_y + 4} Z"
    )

    clove_right = (
        f"M{cx},{top_y + 4} "
        f"C{cx + 3},{top_y + 12} {cx + 18},{mid_y - 18} {cx + 20},{mid_y - 2} "
        f"C{cx + 20},{mid_y + 6} {cx + 12},{bottom_y - 16} {cx + 4},{bottom_y - 8} "
        f"C{cx + 6},{bottom_y - 20} {cx + 14},{mid_y} {cx + 14},{mid_y - 8} "
        f"C{cx + 14},{mid_y - 20} {cx + 2},{top_y + 10} {cx},{top_y + 4} Z"
    )

    return IconPaths(main, clove_left, clove_right, center_crease, shade, highlight)


def generate_defs(color: Color) -> str:
    return f'''  <defs>
    <radialGradient id="baseGradient" cx="35%" cy="30%" r="70%" fx="25%" fy="20%">
      <stop offset="0%" stop-color="{color.highlight}"/>
      <stop offset="45%" stop-color="{color.fill}"/>
      <stop offset="100%" stop-color="{color.deep_shadow}"/>
    </radialGradient>
  </defs>'''


def generate_icon_elements(paths: IconPaths, color: Color, outline_width: float = 2) -> str:
    accent = "#D49560"
    return f'''    <path id="Background" d="{paths.main}" fill="{color.fill}"/>
    <path id="CloveLeft" d="{paths.clove_left}" fill="{accent}"/>
    <path id="CloveRight" d="{paths.clove_right}" fill="{accent}"/>
    <path id="CenterCrease" d="{paths.center_crease}" stroke="{color.outline}" stroke-width="1" stroke-opacity="0.4" fill="none" stroke-linecap="round"/>
    <path id="BottomEdge" d="{paths.shade}" fill="{accent}"/>
    <path id="Outline" d="{paths.main}" fill="none" stroke="{color.outline}" stroke-width="{outline_width}"/>'''


def round_path(path: str, precision: int = 2) -> str:
    import re
    def round_match(m):
        return str(round(float(m.group(0)), precision)).rstrip('0').rstrip('.')
    return re.sub(r'-?\d+\.\d+', round_match, path)


def generate_letter_paths(text: str, font_path: Path, scale: float = 0.08, weight: int = 400) -> tuple[list[str], float, float]:
    font = TTFont(font_path)

    if "fvar" in font:
        from fontTools.varLib.mutator import instantiateVariableFont
        font = instantiateVariableFont(font, {"wght": weight})

    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    descender = font["hhea"].descent * scale

    paths = []
    x_offset = 0

    for char in text:
        glyph_name = cmap.get(ord(char))
        if glyph_name:
            glyph = glyph_set[glyph_name]
            pen = SVGPathPen(glyph_set)
            transform_pen = TransformPen(pen, (scale, 0, 0, -scale, x_offset * scale, 0))
            glyph.draw(transform_pen)
            path = pen.getCommands()
            if path:
                paths.append(round_path(path))
            x_offset += glyph.width

    return paths, round(x_offset * scale, 1), descender


def generate_text_elements(letter_paths: list[str], color: Color) -> str:
    combined = " ".join(letter_paths)
    # plain wordmark — standard Outfit-700 glyphs, same as the site text (no cel shadow/outline)
    return f'    <path d="{combined}" fill="{color.text}"/>'


def generate_icon(
    width: int = 80,
    height: int = 80,
    color: Color = DEFAULT_COLOR,
    outline_width: float = 2,
) -> str:
    paths = generate_icon_paths()
    defs = generate_defs(color)
    elements = generate_icon_elements(paths, color, outline_width)
    cx, cy = width / 2, height / 2

    return f'''<svg id="Shallot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <title>shallot icon</title>

{defs}

  <g transform="rotate(35 {cx} {cy})">
{elements}
  </g>
</svg>'''


def generate_logo(
    icon_size: int = 80,
    text_scale: float = 0.058,
    color: Color = DEFAULT_COLOR,
    gap: float = 6,
    trailing: float = 10,
) -> str:
    font_path = Path(__file__).parent / "font.ttf"
    letter_paths, text_width, descender = generate_letter_paths("shallot", font_path, text_scale, weight=700)

    # the 35°-rotated onion fills the 80-box out to ~x=64; start the text just past that,
    # not at the box edge, so the wordmark sits close to the icon
    icon_visible_right = 64
    text_x = round(icon_visible_right + gap)
    text_y = round(icon_size - 6 + descender)
    total_width = round(text_x + text_width + trailing)

    paths = generate_icon_paths()
    defs = generate_defs(color)
    icon_elements = generate_icon_elements(paths, color)
    text_elements = generate_text_elements(letter_paths, color)

    return f'''<svg id="ShallotLogo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width} {icon_size}">
  <title>shallot logo</title>

{defs}

  <g id="Icon" transform="rotate(35 40 40)">
{icon_elements}
  </g>

  <g id="Text" transform="translate({text_x} {text_y})">
{text_elements}
  </g>
</svg>'''


def svg_to_png(svg_path: Path, png_path: Path, scale: int = 1):
    import cairosvg
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)


if __name__ == "__main__":
    base = Path(__file__).parent

    icon_svg = generate_icon()
    icon_output = base / "icon.svg"
    icon_output.write_text(icon_svg)
    print(f"Written to {icon_output}")

    svg_to_png(icon_output, base / "icon-1024.png", scale=12)
    print(f"Written to {base / 'icon-1024.png'}")

    logo_svg = generate_logo()
    logo_output = base / "logo.svg"
    logo_output.write_text(logo_svg)
    print(f"Written to {logo_output}")

    svg_to_png(logo_output, base / "logo-1024.png", scale=12)
    print(f"Written to {base / 'logo-1024.png'}")
