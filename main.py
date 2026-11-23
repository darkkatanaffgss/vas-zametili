import argparse
import math
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


COLS = 160
ROWS = 56
TARGET_FPS = 40

ASCII_RAMP = np.array(list(" .':,;~-=+*oO0#MW%@"))
RAMP_LEN = len(ASCII_RAMP)


@dataclass(frozen=True)
class Theme:
    name: str
    stops: Tuple[Tuple[float, Tuple[int, int, int]], ...]
    hud_fg: str


THEMES: Tuple[Theme, ...] = (
    Theme(
        name="Skyrim Silver",
        stops=(
            (0.00, (16, 18, 20)),
            (0.22, (74, 76, 74)),
            (0.56, (172, 170, 162)),
            (0.86, (224, 222, 212)),
            (1.00, (248, 245, 234)),
        ),
        hud_fg="#585752",
    ),
    Theme(
        name="Detected Crimson",
        stops=(
            (0.00, (26, 6, 8)),
            (0.22, (122, 24, 18)),
            (0.56, (218, 70, 46)),
            (0.86, (250, 158, 126)),
            (1.00, (255, 232, 218)),
        ),
        hud_fg="#70302a",
    ),
    Theme(
        name="Nordic Frost",
        stops=(
            (0.00, (8, 18, 26)),
            (0.22, (42, 94, 118)),
            (0.56, (118, 198, 228)),
            (0.86, (198, 238, 252)),
            (1.00, (240, 252, 255)),
        ),
        hud_fg="#3e6273",
    ),
    Theme(
        name="Dwemer Gold",
        stops=(
            (0.00, (22, 16, 6)),
            (0.22, (112, 80, 24)),
            (0.56, (210, 162, 56)),
            (0.86, (246, 214, 130)),
            (1.00, (255, 244, 202)),
        ),
        hud_fg="#6e5a2e",
    ),
)


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def sdf_segment(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def sdf_elliptical_arc(
    px: np.ndarray,
    py: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    a_start: float,
    a_end: float,
) -> np.ndarray:
    ang = np.arctan2((py - cy) / max(ry, 1e-6), (px - cx) / max(rx, 1e-6))
    mid = 0.5 * (a_start + a_end)
    half = 0.5 * (a_end - a_start)
    diff = (ang - mid + math.pi) % (2.0 * math.pi) - math.pi
    clamped = mid + np.clip(diff, -half, half)
    return np.hypot(px - (cx + rx * np.cos(clamped)), py - (cy + ry * np.sin(clamped)))


GlyphPrimitive = Tuple[str, Tuple[float, ...]]

GLYPH_DEFS: Dict[str, List[GlyphPrimitive]] = {
    "В": [
        ("seg", (-0.33, -0.50, -0.33, 0.50)),
        ("seg", (-0.33, -0.50, 0.03, -0.50)),
        ("seg", (-0.33, -0.02, 0.05, -0.02)),
        ("seg", (-0.33, 0.50, 0.06, 0.50)),
        ("arc", (0.03, -0.26, 0.29, 0.24, -0.5 * math.pi, 0.5 * math.pi)),
        ("arc", (0.06, 0.24, 0.32, 0.26, -0.5 * math.pi, 0.5 * math.pi)),
    ],
    "А": [
        ("seg", (-0.37, 0.50, 0.00, -0.50)),
        ("seg", (0.00, -0.50, 0.37, 0.50)),
        ("seg", (-0.23, 0.16, 0.23, 0.16)),
    ],
    "С": [
        ("arc", (0.05, 0.00, 0.39, 0.50, 0.24 * math.pi, 1.76 * math.pi)),
    ],
    "З": [
        ("arc", (-0.02, -0.26, 0.35, 0.24, -0.80 * math.pi, 0.48 * math.pi)),
        ("seg", (-0.06, -0.02, 0.02, -0.02)),
        ("arc", (0.00, 0.24, 0.36, 0.26, -0.48 * math.pi, 0.80 * math.pi)),
    ],
    "М": [
        ("seg", (-0.35, 0.50, -0.34, -0.50)),
        ("seg", (-0.34, -0.50, 0.00, 0.40)),
        ("seg", (0.00, 0.40, 0.34, -0.50)),
        ("seg", (0.34, -0.50, 0.35, 0.50)),
    ],
    "Е": [
        ("seg", (-0.30, -0.50, -0.30, 0.50)),
        ("seg", (-0.30, -0.50, 0.29, -0.50)),
        ("seg", (-0.30, -0.02, 0.19, -0.02)),
        ("seg", (-0.30, 0.50, 0.29, 0.50)),
    ],
    "Т": [
        ("seg", (-0.31, -0.50, 0.31, -0.50)),
        ("seg", (0.00, -0.50, 0.00, 0.50)),
    ],
    "И": [
        ("seg", (-0.32, -0.50, -0.32, 0.50)),
        ("seg", (-0.32, 0.50, 0.32, -0.50)),
        ("seg", (0.32, -0.50, 0.32, 0.50)),
    ],
    "Л": [
        ("seg", (-0.06, -0.50, 0.32, -0.50)),
        ("seg", (0.32, -0.50, 0.32, 0.50)),
        ("seg", (-0.06, -0.50, -0.14, 0.18)),
        ("seg", (-0.14, 0.18, -0.36, 0.50)),
    ],
}


def build_text_glyphs(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    text: str = "ВАС ЗАМЕТИЛИ",
    center_y: float = 0.275,
    letter_h: float = 0.242,
    letter_w: float = 0.095,
    advance: float = 0.118,
    space_advance: float = 0.072,
) -> Tuple[List[Tuple[float, int, int, np.ndarray]], np.ndarray]:
    """Precompute tight column-sliced glyph layers plus a cached full-text layer."""
    homoglyphs = {"B": "В", "A": "А", "C": "С", "E": "Е", "T": "Т", "M": "М", "H": "И"}
    normalized_chars = [homoglyphs.get(c, c) for c in text.upper()]
    widths = [space_advance if ch == " " else advance for ch in normalized_chars]
    total_w = sum(widths) - (advance - letter_w)
    cursor = -0.5 * total_w + 0.5 * letter_w
    xs_1d = grid_x[0]
    pad = letter_w * 0.95

    glyph_slices: List[Tuple[float, int, int, np.ndarray]] = []
    full_text = np.zeros_like(grid_x, dtype=np.float32)

    for ch in normalized_chars:
        if ch == " ":
            cursor += space_advance
            continue
        if ch not in GLYPH_DEFS:
            cursor += advance
            continue

        cx = cursor
        c0 = int(np.searchsorted(xs_1d, cx - pad, side="left"))
        c1 = int(np.searchsorted(xs_1d, cx + pad, side="right"))
        c0 = max(0, c0)
        c1 = min(grid_x.shape[1], c1)

        u = (grid_x[:, c0:c1] - cx) / letter_w
        v = (grid_y[:, c0:c1] - center_y) / letter_h

        dist = np.full_like(u, 10.0, dtype=np.float32)
        for kind, params in GLYPH_DEFS[ch]:
            d = sdf_segment(u, v, *params) if kind == "seg" else sdf_elliptical_arc(u, v, *params)
            np.minimum(dist, d, out=dist)

        sdf = dist * (math.sqrt(letter_w * letter_h) * 0.62)
        core = np.clip(1.0 - np.maximum(0.0, sdf - 0.0072) / 0.0078, 0.0, 1.0)
        bloom = np.exp(-sdf / 0.012) * 0.15
        bloom = np.where(bloom > 0.05, bloom, 0.0)
        layer = np.clip(core * 0.96 + bloom, 0.0, 1.0).astype(np.float32)

        glyph_slices.append((cx, c0, c1, layer))
        np.maximum(full_text[:, c0:c1], layer, out=full_text[:, c0:c1])
        cursor += advance

    return glyph_slices, full_text


class Scene:
    def __init__(self, cols: int = COLS, rows: int = ROWS, cell_aspect: float = 2.0) -> None:
        self.cols = cols
        self.rows = rows
        self.cell_aspect = cell_aspect

        xs = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
        y_half = (rows * cell_aspect) / cols
        ys = np.linspace(-y_half, y_half, rows, dtype=np.float32)
        self.gx, self.gy = np.meshgrid(xs, ys)
        self.y_half = y_half

        self.cy = -0.205
        self.rx = 0.740
        self.max_h = 0.258
        self.iris_r = 0.236
        self.pupil_r = 0.019

        # Precompute static coordinate helpers
        self.x = self.gx
        self.y = (self.gy - self.cy).astype(np.float32)
        self.x2 = self.x * self.x
        self.in_span = np.abs(self.x) <= self.rx
        self.d_tips = np.minimum(np.hypot(self.x - self.rx, self.y), np.hypot(self.x + self.rx, self.y))
        self.tip_flare = (
            np.exp(-((self.x - self.rx) ** 2 + self.y * self.y) / 0.0022)
            + np.exp(-((self.x + self.rx) ** 2 + self.y * self.y) / 0.0022)
        ).astype(np.float32) * 0.22

        self.r_center = np.hypot(self.x, self.y)
        self.ang_center = np.arctan2(self.y, self.x)
        self.ellip_r = np.hypot(self.x / 1.55, self.y)

        # Ambient background mist behind the focal silhouette
        self.aura_base = (
            np.exp(-(self.x2 / 0.72 + (self.y * self.y) / 0.22) * 1.65) * 0.085
        ).astype(np.float32)
        self.vignette = np.clip(
            1.14 - 0.28 * (self.x2 + (self.gy / y_half) ** 2),
            0.55,
            1.0,
        ).astype(np.float32)
        self.shimmer_phase = (self.gx * 24.0 + self.gy * 16.0).astype(np.float32)

        # Precompute lock-on crosshair ticks around the center
        d_cross_h = np.minimum(
            sdf_segment(self.x, self.y, -0.94, 0.0, -0.82, 0.0),
            sdf_segment(self.x, self.y, 0.82, 0.0, 0.94, 0.0),
        )
        d_cross_v = np.minimum(
            sdf_segment(self.x, self.y, 0.0, -0.36, 0.0, -0.30),
            sdf_segment(self.x, self.y, 0.0, 0.30, 0.0, 0.36),
        )
        d_cross = np.minimum(d_cross_h, d_cross_v)
        self.crosshair_glow = (np.exp(-d_cross / 0.011) * 0.55).astype(np.float32)

        # Embers / dust particles scattered in O(count) per frame
        rng = np.random.default_rng(42)
        self.dust_count = 90
        self.dust_x = rng.uniform(0.0, cols - 1.0, self.dust_count)
        self.dust_y = rng.uniform(0.0, rows - 1.0, self.dust_count)
        self.dust_vx = rng.uniform(-1.1, 1.1, self.dust_count)
        self.dust_vy = rng.uniform(-1.6, -0.35, self.dust_count)
        self.dust_phase = rng.uniform(0.0, 2.0 * math.pi, self.dust_count)
        self.dust_amp = rng.uniform(0.18, 0.38, self.dust_count)

        self.glyph_slices, self.full_text_layer = build_text_glyphs(self.gx, self.gy)

        self.manual_blink_t = -10.0
        self.pulse_times: List[float] = []

    def trigger_blink_pulse(self, t: float) -> None:
        self.manual_blink_t = t
        self.pulse_times.append(t)
        self.pulse_times = [pt for pt in self.pulse_times if t - pt < 2.2]

    def reset(self) -> None:
        self.manual_blink_t = -10.0
        self.pulse_times.clear()

    def _timeline(self, t: float) -> Tuple[float, float, float, float, float]:
        open_amt = ease_out_cubic((t - 1.1) / 1.55)

        if t > 8.0:
            cycle = (t - 8.0) % 6.5
            if cycle < 0.26:
                b = math.sin((cycle / 0.26) * math.pi)
                open_amt *= 1.0 - 0.92 * (b ** 1.4)

        dt_manual = t - self.manual_blink_t
        if 0.0 <= dt_manual < 0.28:
            b = math.sin((dt_manual / 0.28) * math.pi)
            open_amt *= 1.0 - 0.94 * (b ** 1.4)

        open_amt = max(0.012, open_amt)

        gaze_x = 0.0
        gaze_y = 0.0
        if 2.75 <= t < 3.65:
            p = ease_in_out_cubic((t - 2.75) / 0.55)
            gaze_x = -0.14 * p
            gaze_y = -0.015 * p
        elif 3.65 <= t < 4.55:
            p = ease_in_out_cubic((t - 3.65) / 0.65)
            gaze_x = -0.14 + 0.28 * p
            gaze_y = -0.015 + 0.025 * math.sin(p * math.pi)
        elif 4.55 <= t < 5.15:
            p = ease_in_out_cubic((t - 4.55) / 0.42)
            gaze_x = 0.14 * (1.0 - p)
            gaze_y = 0.0

        flash_open = math.exp(-((t - 2.45) / 0.35) ** 2) * 0.35
        flash_lock = math.exp(-((t - 5.15) / 0.40) ** 2) * 0.58
        alert_flash = flash_open + flash_lock

        text_reveal = clamp((t - 5.25) / 1.75)
        return open_amt, gaze_x, gaze_y, alert_flash, text_reveal

    def _scatter_dust(self, buf: np.ndarray, t: float) -> None:
        xs = (self.dust_x + self.dust_vx * t) % self.cols
        ys = (self.dust_y + self.dust_vy * t) % self.rows
        tw = 0.45 + 0.55 * np.sin(t * 2.2 + self.dust_phase)

        for i in range(self.dust_count):
            cx = int(xs[i])
            cy = int(ys[i])
            amp = float(self.dust_amp[i] * tw[i])
            if amp > buf[cy, cx]:
                buf[cy, cx] = amp
            # Faint rising ember trail one cell below
            ny = (cy + 1) % self.rows
            trail = amp * 0.48
            if trail > buf[ny, cx]:
                buf[ny, cx] = trail

    def compute_frame(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        open_amt, gaze_x, gaze_y, alert_flash, text_reveal = self._timeline(t)

        h = max(0.008, self.max_h * open_amt)
        rc = (self.rx * self.rx + h * h) / (2.0 * h)
        yc = h - rc

        y_upper = yc + np.sqrt(np.maximum(0.0, rc * rc - self.x2))
        y_lower = -y_upper

        d_upper_arc = np.abs(np.hypot(self.x, self.y - yc) - rc)
        d_lower_arc = np.abs(np.hypot(self.x, self.y + yc) - rc)
        d_lid = np.where(self.in_span, np.minimum(d_upper_arc, d_lower_arc), self.d_tips)

        open_margin = np.where(self.in_span, np.minimum(y_upper - self.y, self.y - y_lower), -self.d_tips)
        inside_mask = np.clip(open_margin / 0.020 + 0.5, 0.0, 1.0)

        slit_fade = 1.0 - smoothstep(1.0, 2.0, t)
        slit_pulse = 0.55 + 0.35 * math.sin(t * 3.2)

        lid_core = np.clip(1.0 - np.maximum(0.0, d_lid - 0.0155) / 0.0135, 0.0, 1.0)
        lid_bloom = np.exp(-d_lid / 0.025) * (0.24 + 0.15 * alert_flash)

        if gaze_x == 0.0 and gaze_y == 0.0:
            r_iris = self.r_center
            ang_iris = self.ang_center
        else:
            ix = self.x - gaze_x
            iy = self.y - gaze_y
            r_iris = np.hypot(ix, iy)
            ang_iris = np.arctan2(iy, ix)

        d_ring = np.abs(r_iris - self.iris_r)
        ring_core = np.clip(1.0 - np.maximum(0.0, d_ring - 0.0145) / 0.013, 0.0, 1.0)
        ring_bloom = np.exp(-d_ring / 0.023) * 0.20

        fibers = 0.5 + 0.5 * np.sin(ang_iris * 28.0 + t * 0.8 + r_iris * 18.0)
        in_iris = np.clip((self.iris_r - 0.02 - r_iris) / 0.02, 0.0, 1.0)
        out_pupil = np.clip((r_iris - self.pupil_r * 2.3) / 0.04, 0.0, 1.0)
        fiber_glow = in_iris * out_pupil * fibers * 0.055 * smoothstep(1.3, 2.6, t)

        d_pupil = np.maximum(0.0, r_iris - self.pupil_r)
        pupil_core = np.clip(1.0 - d_pupil / 0.015, 0.0, 1.0)
        pupil_bloom = np.exp(-d_pupil / 0.030) * (0.28 + 0.26 * alert_flash)

        reveal_main = smoothstep(0.15, 1.15, t)
        main_layer = (
            lid_core * (0.93 + 0.07 * alert_flash)
            + lid_bloom
            + self.tip_flare * smoothstep(1.4, 2.4, t)
            + inside_mask * (ring_core * 0.90 + ring_bloom + pupil_core * 0.98 + pupil_bloom + fiber_glow)
        )
        main_layer *= (0.45 * slit_fade * slit_pulse + (1.0 - slit_fade)) * reveal_main

        # Transient lock-on crosshair accent around t = 4.9 .. 6.6s
        cross_env = math.exp(-((t - 5.25) / 0.65) ** 2)
        if cross_env > 0.02:
            main_layer += self.crosshair_glow * cross_env

        # Text layer: cached when fully revealed, sliced per glyph during reveal
        if text_reveal >= 1.0:
            text_layer = self.full_text_layer
        elif text_reveal > 0.0:
            text_layer = np.zeros_like(self.gx, dtype=np.float32)
            n_glyphs = len(self.glyph_slices)
            for idx, (cx, c0, c1, glyph_buf) in enumerate(self.glyph_slices):
                order = abs(cx) / 0.72 * 0.45 + (idx / n_glyphs) * 0.25
                local_p = clamp((text_reveal * 1.35 - order) / 0.45)
                if local_p <= 0.0:
                    continue
                eased = ease_out_cubic(local_p)
                flash = 1.0 + math.exp(-((local_p - 0.55) / 0.28) ** 2) * 0.24
                np.maximum(text_layer[:, c0:c1], glyph_buf * (eased * flash), out=text_layer[:, c0:c1])
        else:
            text_layer = None

        # Background particles & soft radial mist
        bg_layer = self.aura_base * (0.85 + 0.15 * math.sin(t * 1.5)) * reveal_main
        self._scatter_dust(bg_layer, t)

        lum = np.maximum(main_layer, bg_layer)
        if text_layer is not None:
            np.maximum(lum, text_layer, out=lum)

        for pt in (2.25, 5.10, *self.pulse_times):
            dt = t - pt
            if 0.0 < dt < 2.0:
                pr = 0.12 + dt * 0.72
                d_ring = np.abs(self.ellip_r - pr)
                fade = (1.0 - dt / 2.0) ** 2
                lum += np.exp(-d_ring / 0.024) * (0.34 * fade)

        breath = 0.96 + 0.04 * math.sin(t * 1.6)
        shimmer = 0.032 * np.sin(self.shimmer_phase - t * 2.6)
        lum = np.where(lum > 0.25, lum * breath + shimmer, lum)
        lum = np.clip(lum * self.vignette, 0.0, 1.0)

        idx = np.clip((np.power(lum, 0.94) * (RAMP_LEN - 1) + 0.35).astype(np.int32), 0, RAMP_LEN - 1)
        return idx, lum


def build_color_lut(theme: Theme, steps: int = 64) -> List[str]:
    stops = theme.stops
    lut: List[str] = []
    for i in range(steps):
        val = i / max(1, steps - 1)
        r, g, b = stops[-1][1]
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]
            t1, c1 = stops[k + 1]
            if t0 <= val <= t1:
                alpha = (val - t0) / max(1e-6, t1 - t0)
                r = int(round(c0[0] + (c1[0] - c0[0]) * alpha))
                g = int(round(c0[1] + (c1[1] - c0[1]) * alpha))
                b = int(round(c0[2] + (c1[2] - c0[2]) * alpha))
                break
        lut.append(f"#{r:02x}{g:02x}{b:02x}")
    return lut


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ВАС ЗАМЕТИЛИ")
        self.root.configure(bg="#000000")
        self.root.resizable(False, False)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        font_size = 9 if (screen_w >= 1200 and screen_h >= 840) else 8

        self.font = tkfont.Font(family="Consolas", size=font_size)
        self.cell_w = max(5, self.font.measure("M"))
        self.cell_h = max(10, self.font.metrics("linespace"))
        self.cell_aspect = self.cell_h / self.cell_w

        self.canvas_w = COLS * self.cell_w
        self.canvas_h = ROWS * self.cell_h

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#000000",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        self.theme_idx = 0
        self.lut_steps = 64
        self.theme_luts = [build_color_lut(th, self.lut_steps) for th in THEMES]
        self.hud_visible = True
        self.paused = False
        self.elapsed = 0.0
        self.last_tick = time.perf_counter()

        self.scene = Scene(COLS, ROWS, cell_aspect=self.cell_aspect)

        self.items: List[int] = [
            self.canvas.create_text(
                (i % COLS) * self.cell_w,
                (i // COLS) * self.cell_h,
                text=" ",
                fill="#000000",
                font=self.font,
                anchor="nw",
            )
            for i in range(COLS * ROWS)
        ]
        self.prev_packed = np.zeros(COLS * ROWS, dtype=np.int32)

        self.hud_item = self.canvas.create_text(
            self.canvas_w // 2,
            self.canvas_h - 10,
            text=self._hud_text(),
            fill=THEMES[self.theme_idx].hud_fg,
            font=("Consolas", 8),
            anchor="s",
        )

        win_x = max(0, (screen_w - self.canvas_w) // 2)
        win_y = max(0, (screen_h - self.canvas_h) // 2 - 20)
        self.root.geometry(f"{self.canvas_w}x{self.canvas_h}+{win_x}+{win_y}")

        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<r>", self._on_restart)
        self.root.bind("<R>", self._on_restart)
        self.root.bind("<space>", self._on_pause)
        self.root.bind("<b>", self._on_blink)
        self.root.bind("<B>", self._on_blink)
        self.root.bind("<m>", self._on_theme)
        self.root.bind("<M>", self._on_theme)
        self.root.bind("<h>", self._on_toggle_hud)
        self.root.bind("<H>", self._on_toggle_hud)

        self._loop()

    def _hud_text(self) -> str:
        theme_name = THEMES[self.theme_idx].name
        state = "PAUSED  |  " if self.paused else ""
        return (
            f"{state}[R] Restart   [Space] Pause   [B] Blink/Pulse   "
            f"[M] Theme: {theme_name}   [H] Hide HUD   [Esc] Exit"
        )

    def _refresh_hud(self) -> None:
        if self.hud_visible:
            self.canvas.itemconfig(
                self.hud_item,
                text=self._hud_text(),
                fill=THEMES[self.theme_idx].hud_fg,
                state="normal",
            )
        else:
            self.canvas.itemconfig(self.hud_item, state="hidden")

    def _on_restart(self, _event: tk.Event) -> None:
        self.elapsed = 0.0
        self.paused = False
        self.scene.reset()
        self._refresh_hud()

    def _on_pause(self, _event: tk.Event) -> None:
        self.paused = not self.paused
        self._refresh_hud()

    def _on_blink(self, _event: tk.Event) -> None:
        self.scene.trigger_blink_pulse(self.elapsed)

    def _on_theme(self, _event: tk.Event) -> None:
        self.theme_idx = (self.theme_idx + 1) % len(THEMES)
        self.prev_packed.fill(-1)
        self._refresh_hud()

    def _on_toggle_hud(self, _event: tk.Event) -> None:
        self.hud_visible = not self.hud_visible
        self._refresh_hud()

    def _render_frame(self) -> None:
        idx_buf, lum_buf = self.scene.compute_frame(self.elapsed)
        lut = self.theme_luts[self.theme_idx]
        lut_max = self.lut_steps - 1

        char_flat = idx_buf.ravel()
        col_flat = np.clip((lum_buf.ravel() * lut_max).astype(np.int32), 0, lut_max)
        packed = np.where(char_flat == 0, 0, (char_flat << 8) | col_flat)

        changed = np.flatnonzero(packed != self.prev_packed)
        if changed.size == 0:
            return

        items = self.items
        ramp = ASCII_RAMP
        itemconfig = self.canvas.itemconfig

        for i in changed.tolist():
            p = int(packed[i])
            if p == 0:
                itemconfig(items[i], text=" ")
            else:
                itemconfig(items[i], text=str(ramp[p >> 8]), fill=lut[p & 0xFF])

        self.prev_packed[:] = packed

    def _loop(self) -> None:
        now = time.perf_counter()
        dt = min(0.08, now - self.last_tick)
        self.last_tick = now
        if not self.paused:
            self.elapsed += dt

        self._render_frame()

        frame_ms = int(1000 / TARGET_FPS)
        spent_ms = int((time.perf_counter() - now) * 1000)
        self.root.after(max(4, frame_ms - spent_ms), self._loop)


def dump_snapshot(t: float) -> str:
    scene = Scene(COLS, ROWS, cell_aspect=2.0)
    idx_buf, _ = scene.compute_frame(t)
    return "\n".join("".join(ASCII_RAMP[row]) for row in idx_buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="ВАС ЗАМЕТИЛИ — ASCII Animation")
    parser.add_argument("--snapshot", type=float, default=None, help="Print ASCII frame at time T")
    args = parser.parse_args()

    if args.snapshot is not None:
        print(dump_snapshot(args.snapshot))
        return

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
