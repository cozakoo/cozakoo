#!/usr/bin/env python3
"""Generate the animated GitHub profile banner.

Run from the repository root:
    python scripts/banner/generate.py

The subject is the profile badge (assets/source/avatar.png): a flat logo on a
transparent background. Its alpha channel is the shape, so the banner draws the
badge itself in 1-bit rather than a photographic portrait. The badge then morphs
into three stack logos and back.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/avatar.png"
ASSETS = ROOT / "assets"
ICONS = ROOT / "icons"

W, H = 1180, 610
LOOP_SECONDS = 14.2
TRAVELLER_COUNT = 900
SEED = 314159

# Phase boundaries in seconds: the badge holds first, then every logo holds for
# two seconds with the morph transitions in between.
TIMELINE = [0, 3.0, 4.3, 6.3, 7.6, 9.6, 10.9, 12.9, 14.2]

# The badge and every logo are drawn on the SAME lattice, so the morph is a
# dot-to-dot transformation instead of a rescale: 300x300 of lit pixels placed
# at the same origin inside the 300x340 visual frame.
LATTICE_W, LATTICE_H = 300, 340
LATTICE_X, LATTICE_Y = 74, 154
SUBJECT_SIZE = 300
SUBJECT_X = LATTICE_X
SUBJECT_Y = LATTICE_Y + (LATTICE_H - SUBJECT_SIZE) // 2

ROWS = [
    ("Subject", "Martín Arcos Vargas"),
    ("Role", "Backend Builder · SOLID Architect"),
    ("Origin", "Puerto Madryn · Chubut · AR"),
    ("Education", "UNPSJB · Lic. Informática"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "Git · Docker · Nginx"),
    ("Core.Lang", "Python · Java · JavaScript"),
    ("Core.Backend", "Django · Spring Boot"),
    ("Core.Frontend", "Bootstrap 5 · HTML/CSS"),
    ("Core.Database", "PostgreSQL · SQLite"),
    ("Core.Infra", "Linux · Docker · Nginx"),
    ("Grid.Mail", "martinarcosvargas2@gmail.com"),
    ("Grid.LinkedIn", "/in/martin-arcos"),
    ("Grid.GitHub", "cozakoo"),
    ("Grid.Portfolio", "arcosvargas.com"),
]

# Stack logos the badge morphs into. Every one of them keeps a solid silhouette
# in its own alpha channel; that is what the dither samples.
LOGOS = [
    ("django", ICONS / "django.png"),
    ("postgres", ICONS / "postgreSQL.png"),
    ("docker", ICONS / "docker.png"),
]

THEMES = {
    "dark": {
        "bg": "#0A101F",
        "panel": "#0D1628",
        "panel2": "#101B30",
        "line": "#25344C",
        "muted": "#8291A8",
        "text": "#DDE7F5",
        "portrait": "#38BDF8",
        "chrome": "#0EA5E9",
        "accent": "#22D3EE",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EDF3F7",
        "line": "#CBD7E1",
        "muted": "#64748B",
        "text": "#172033",
        "portrait": "#0369A1",
        "chrome": "#0284C7",
        "accent": "#0EA5E9",
        "shadow": "#AAB7C4",
    },
}


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg diffusion; True means a lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def subject_points(theme: str) -> np.ndarray:
    """Sample the badge alpha into banner coordinates.

    The badge is a flat logo on transparency, so the alpha channel *is* the
    shape: interior pixels are fully opaque and the ring around the monogram
    is semi-transparent, which the dither renders as a stippled circle. The
    same selection is used for both themes — only the ink colour changes — and
    the subsample is seeded so switching the OS theme does not reshuffle the
    dots.
    """
    source = Image.open(SOURCE).convert("RGBA")
    scale = SUBJECT_SIZE / source.width
    width = SUBJECT_SIZE
    height = max(1, round(source.height * scale))
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    alpha = np.asarray(resized.getchannel("A"), dtype=np.float32) / 255.0

    bits = floyd_steinberg(alpha * 255.0)
    ys, xs = np.where(bits)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points = np.column_stack((SUBJECT_X + xs, SUBJECT_Y + ys)).astype(np.float32)
    if len(points) > 18000:
        picker = np.random.default_rng(SEED)
        points = points[picker.choice(len(points), 18000, replace=False)]
    return points


def load_logo(path: Path) -> Image.Image:
    """Normalise a brand PNG to the 400x400 sampling canvas.

    The canvas keeps the silhouette centred and scaled the same way for every
    logo, so the frame mapping below stays valid regardless of source size.
    """
    img = Image.open(path).convert("RGBA")
    bbox = img.getchannel("A").point(lambda v: 255 if v > 127 else 0).getbbox()
    if bbox:
        img = img.crop(bbox)
    scale = min(400 / img.width, 400 / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    canvas.paste(img, ((400 - size[0]) // 2, (400 - size[1]) // 2), img)
    return canvas


def logo_dither(image: Image.Image) -> np.ndarray:
    """Dither a logo onto the badge lattice and return its lit pixels.

    The logo is composited on white and reduced to a shape mask first: a brand
    colour is never pure black, and dithering a mid tone gives a 50% stipple
    where the badge is solid. Masking to ink and keeping only the anti-aliased
    edge as a gradient makes both speak the same dot language on the same
    lattice, so the morph is a dot-to-dot transformation instead of a rescale.
    """
    flat = Image.new("RGBA", image.size, "white")
    flat.alpha_composite(image)
    gray = np.asarray(flat.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray
    shape = Image.fromarray(np.uint8((ink > 0.35) * 255)).resize(
        (SUBJECT_SIZE, SUBJECT_SIZE), Image.Resampling.LANCZOS
    )
    return floyd_steinberg(np.asarray(shape, dtype=np.float32))


def logo_points(bits: np.ndarray, rng: np.random.Generator,
                count: int) -> np.ndarray:
    """Sample the dithered logo, stratified so the shape still reads.

    The pick is stratified over a grid instead of purely random, so the sparse
    travellers describe the logo rather than clumping in one corner of it.
    """
    ys, xs = np.where(bits)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    side = int(math.ceil(math.sqrt(count)))
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    gy = np.clip(((ys - y0) / max(y1 - y0, 1) * (side - 1)).astype(int), 0, side - 1)
    gx = np.clip(((xs - x0) / max(x1 - x0, 1) * (side - 1)).astype(int), 0, side - 1)
    perm = rng.permutation(len(xs))
    _, first = np.unique(gy[perm] * side + gx[perm], return_index=True)
    chosen = perm[first]
    if len(chosen) < count:
        extra = rng.choice(len(xs), count - len(chosen), replace=False)
        chosen = np.concatenate((chosen, extra))
    return np.column_stack((SUBJECT_X + xs[chosen], SUBJECT_Y + ys[chosen])).astype(
        np.float32
    )


def logo_layer_points(bits: np.ndarray) -> np.ndarray:
    """Return the dithered logo shape as banner coordinates.

    The travellers alone are too sparse to read as a logo, so the shape is also
    drawn as a layer that fades in while its morph frame is on screen.
    """
    ys, xs = np.where(bits)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack((SUBJECT_X + xs, SUBJECT_Y + ys)).astype(np.float32)


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Stable monospace width used both for textLength and leader placement."""
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_shapes: dict[str, np.ndarray],
    logo_layers: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]

    order = [name for name, _ in LOGOS]
    morphed = []
    current = source
    for name in order:
        current = transport(current, logo_shapes[name][:n])
        morphed.append(current)

    # Explicit uneven phase boundaries: 3.0 badge, 2.0 per logo, and the
    # transitions in between add up to 14.2 seconds. Every animated element
    # uses these key times with three decimals so the travellers, the badge
    # and the logo layers all run on the same clock.
    key_times = ";".join(f"{v / LOOP_SECONDS:.3f}" for v in TIMELINE)
    # Returning each traveller to its exact starting coordinate keeps the
    # repeat boundary seamless. All logo-to-logo morphs use optimal transport.
    frames = [source, source]
    for shape in morphed:
        frames.extend([shape, shape])
    frames.append(source)
    opacity_values = "0;0;1;1;1;1;1;1;0"

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Martin's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a dithered badge and '
        "Django, PostgreSQL and Docker silhouettes.</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">profile.sh --live</text>',
        # Left visual frame.
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×300 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        # Loop layer stays visible at t=0 so camo/static first frames still show
        # the badge. Intro duplicate below shimmers on top, then hands off.
        '<g opacity="1">',
    ]

    # Dense badge drift: 94 independently noisy bands moving toward the first
    # logo centroid.
    first_centroid = logo_shapes[order[0]][:n].mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 4, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (first_centroid - centroid) * 0.18 + noise[band]
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="0 0;0 0;{num(delta[0])} {num(delta[1])};'
            f'{num(delta[0])} {num(delta[1])};0 0;0 0;0 0;0 0;0 0"/>'
            f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            'values=".94;.94;0;0;0;0;0;0;.94"/></path>'
        )

    # Silhouette layers: each logo fades in AFTER its particles have finished
    # drawing the dotted shape, so the sequence reads "the dots make the form,
    # then the image appears" instead of the fill showing up mid-flight.
    for index, name in enumerate(order):
        pts = logo_layers[name]
        if not len(pts):
            continue
        start = TIMELINE[2 + index * 2]
        end = TIMELINE[3 + index * 2]
        delay = 0.15
        fade = 0.5
        key = ";".join(
            f"{value:.3f}"
            for value in (
                0,
                (start + delay) / LOOP_SECONDS,
                (start + delay + fade) / LOOP_SECONDS,
                end / LOOP_SECONDS,
                (end + fade) / LOOP_SECONDS,
                1,
            )
        )
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" keyTimes="{key}" values="0;0;1;1;0;0"/></path>'
        )

    # Optimal-transport travellers, represented as tiny path squares (never glyphs).
    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )
    parts.append("</g>")

    parts.extend(
        [
            "</g>",
            # Small frame telemetry.
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · FS/SERPENTINE</text>',
            # Right information panel.
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
            # LIVE badge and handle pill.
            '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A">'
            '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/>'
            '</circle></g>',
            '<text x="927" y="111" fill="#FF4D5A" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" '
            'font-weight="700">LIVE</text>',
            f'<rect x="982" y="94" width="146" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1055" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" '
            'font-weight="700">@cozakoo</text>',
        ]
    )

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend(
            [
                f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">'
                f"{html.escape(label)}</text>",
                f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
                f'fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
                f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" '
                f'fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
                f'font-size="14" textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(value)}</text>",
            ]
        )
        row_y += 23

    parts.extend(
        [
            f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
            f'<text x="491" y="548" fill="{t["accent"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "● ALL SYSTEMS NOMINAL</text>",
            f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "UTC-3 · PATAGONIA NODE</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source badge: {SOURCE}")
    for _, path in LOGOS:
        if not path.exists():
            raise SystemExit(f"Missing logo: {path}")

    ASSETS.mkdir(parents=True, exist_ok=True)
    logos = [(name, load_logo(path)) for name, path in LOGOS]

    for theme in THEMES:
        # One seed for every theme: the two banners must be the same drawing in
        # two palettes, so switching the OS theme does not reshuffle the dots.
        rng = np.random.default_rng(SEED)
        portrait = subject_points(theme)
        masks = {name: logo_dither(image) for name, image in logos}
        sampled = {
            name: logo_points(mask, rng, TRAVELLER_COUNT)
            for name, mask in masks.items()
        }
        layers = {name: logo_layer_points(mask) for name, mask in masks.items()}
        svg = render_svg(theme, portrait, sampled, layers, rng)
        output = ASSETS / f"banner-{theme}.v1.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portrait):,} badge dots, "
            f"{TRAVELLER_COUNT} travellers"
        )


if __name__ == "__main__":
    main()
