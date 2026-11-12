#!/usr/bin/env python
"""Regenerate every figure of the paper from the JSON result files.

This script only reads JSON result files from the repository root (plus system
fonts) and writes PNGs into paper/figures/.  It never imports mlx / mlx_vlm,
never loads a model or a patch, and does not use the network.

matplotlib is deliberately not required: all drawing is done with Pillow.
Run from the repository root:

    .venv/bin/python make_figures.py

Every number in every panel is read from a JSON file at run time; the source
file and the plotted values are printed to stdout.  A missing result file is
reported as a warning and the affected panel (or bar/point) is skipped.
"""

from __future__ import annotations

import json
import math
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - environment check
    raise SystemExit(
        "Pillow is required (it is in requirements.txt): %s" % exc
    )


# --------------------------------------------------------------------------- #
# paths and constants
# --------------------------------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
FIGDIR = os.path.join(ROOT, "paper", "figures")

DPI = 150
SS = 3  # supersampling factor; the image is downsampled at the end

RES_FLIP = "res_flip_L8-12-16-20-24_it800_n300.json"
RES_DENSE = "res_dense_ceiling_L8-12-16-20-24.json"
RES_COMPILE = "res_compile_zs_L8-12-16-20-24.json"
RES_READOUT4 = "res_readout_n300_dev_sup0.json"
RES_READOUT104 = "res_readout_n1000_dev_sup100.json"
RES_TP_PLAIN = "res_taskpatch_summary_plain_n300_dev.json"
RES_TP_GATED = "res_taskpatch_summary_gated_n300_dev.json"
RES_GATED3 = "res_gated_v3.json"
RES_GATED6 = "res_gated_v6.json"

WHITE = (255, 255, 255)
INK = (30, 30, 30)
AXIS = (55, 55, 55)
MUTED = (105, 105, 105)
FAINT = (150, 150, 150)
GRID = (230, 230, 230)
GAPLINE = (205, 205, 205)
HALO = (255, 255, 255)

C_REF = (105, 105, 105)
C_REF_LIGHT = (168, 168, 168)
C_WEIGHT = (57, 106, 160)
C_SIDE = (70, 132, 88)
C_READOUT = (190, 120, 38)
C_GATED = (116, 86, 158)
C_SUMMARY = (44, 120, 120)

TICK_PT = 7.2
XLABEL_PT = 8.0
ANNOT_PT = 7.0
SMALL_PT = 6.4
PANEL_PT = 8.6

FONT_REG = None
FONT_BOLD = None
_FONT_CACHE = {}

for _reg, _bold in (
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    (
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ),
):
    if os.path.isfile(_reg):
        FONT_REG = _reg
        FONT_BOLD = _bold if os.path.isfile(_bold) else _reg
        break


# --------------------------------------------------------------------------- #
# data loading / small helpers
# --------------------------------------------------------------------------- #

_JSON_CACHE = {}
_MISSING_WARNED = set()


def warn(msg):
    print("WARNING: " + msg, file=sys.stderr)


def load_json(name):
    """Load a result JSON from the repo root, caching it.  None if missing."""
    if name in _JSON_CACHE:
        return _JSON_CACHE[name]
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        path = os.path.join(ROOT, "results", name)
    if not os.path.isfile(path):
        if name not in _MISSING_WARNED:
            warn("missing result file %s -- dependent panels are skipped" % name)
            _MISSING_WARNED.add(name)
        _JSON_CACHE[name] = None
    else:
        with open(path, "r", encoding="utf-8") as fh:
            _JSON_CACHE[name] = json.load(fh)
    return _JSON_CACHE[name]


def entry(node, *keys):
    """Walk nested dict keys; return None when any key is absent."""
    cur = node
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def ci_tuple(node):
    """Return (estimate, lo, hi) from a [est, lo, hi] JSON list, else None."""
    if (
        isinstance(node, (list, tuple))
        and len(node) == 3
        and all(isinstance(x, (int, float)) for x in node)
    ):
        return float(node[0]), float(node[1]), float(node[2])
    return None


def point_val(node):
    """Return a scalar point estimate (scalar or first element of a list)."""
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
        return float(node[0])
    return None


def nice_ticks(lo, hi, target=6):
    if not (hi > lo):
        return [lo]
    span = hi - lo
    raw = span / max(1, target)
    p = 10.0 ** math.floor(math.log10(raw))
    m = raw / p
    if m <= 1.0:
        step = p
    elif m <= 2.0:
        step = 2.0 * p
    elif m <= 5.0:
        step = 5.0 * p
    else:
        step = 10.0 * p
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def tint(color, f=0.45):
    """Mix a colour with white; f=1 keeps the colour, f=0 gives white."""
    return tuple(int(round(255 - (255 - c) * f)) for c in color)


def fmt1(v):
    return "%.1f" % v


def fmt2(v):
    return "%.2f" % v


def kb(nbytes):
    return "%.1f KB" % (float(nbytes) / 1024.0)


# --------------------------------------------------------------------------- #
# drawing primitives
# --------------------------------------------------------------------------- #


def _font(size_pt, bold=False):
    key = (round(float(size_pt), 2), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    px = max(8, int(round(size_pt * DPI / 72.0 * SS)))
    try:
        if FONT_REG:
            f = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, px)
        else:
            f = ImageFont.load_default(size=px)
    except Exception:
        f = ImageFont.load_default(size=px)
    _FONT_CACHE[key] = f
    return f


class Canvas:
    def __init__(self, width_in, height_in):
        self.W = int(round(width_in * DPI))
        self.H = int(round(height_in * DPI))
        self.img = Image.new("RGB", (self.W * SS, self.H * SS), WHITE)
        self.d = ImageDraw.Draw(self.img)

    # -- geometry ---------------------------------------------------------
    def line(self, x0, y0, x1, y1, color=INK, w=1.0, dash=None):
        width = max(1, int(round(w * SS)))
        if dash is None:
            self.d.line(
                [(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=color, width=width
            )
            return
        on, off = dash
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux, uy = dx / length, dy / length
        t = 0.0
        while t < length:
            t2 = min(t + on, length)
            self.d.line(
                [
                    ((x0 + ux * t) * SS, (y0 + uy * t) * SS),
                    ((x0 + ux * t2) * SS, (y0 + uy * t2) * SS),
                ],
                fill=color,
                width=width,
            )
            t += on + off

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1.0):
        width = max(1, int(round(w * SS)))
        self.d.rectangle(
            [x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=fill, outline=outline, width=width
        )

    def poly(self, points, fill=None, outline=None, w=1.0):
        width = max(1, int(round(w * SS)))
        self.d.polygon(
            [(x * SS, y * SS) for x, y in points], fill=fill, outline=outline, width=width
        )

    def circle(self, cx, cy, r, fill=None, outline=None, w=1.0):
        width = max(1, int(round(w * SS)))
        self.d.ellipse(
            [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
            fill=fill,
            outline=outline,
            width=width,
        )

    # -- text -------------------------------------------------------------
    def text(self, x, y, s, size=ANNOT_PT, color=INK, bold=False, anchor="lm", halo=False):
        f = _font(size, bold)
        if halo:
            for dx, dy in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ):
                self.d.text(
                    ((x + dx * 1.1) * SS, (y + dy * 1.1) * SS),
                    s,
                    font=f,
                    fill=HALO,
                    anchor=anchor,
                )
        self.d.text((x * SS, y * SS), s, font=f, fill=color, anchor=anchor)

    def twidth(self, s, size=ANNOT_PT, bold=False):
        return self.d.textlength(s, font=_font(size, bold)) / SS

    def vtext(self, cx, cy, s, size=XLABEL_PT, color=INK, bold=False):
        """Rotated (90 deg) text centred at (cx, cy)."""
        f = _font(size, bold)
        bbox = self.d.textbbox((0, 0), s, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3 * SS
        tmp = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.text((pad - bbox[0], pad - bbox[1]), s, font=f, fill=color)
        tmp = tmp.rotate(90, expand=True)
        self.img.paste(
            tmp,
            (int(cx * SS - tmp.width / 2), int(cy * SS - tmp.height / 2)),
            tmp,
        )

    # -- markers ----------------------------------------------------------
    def marker(self, x, y, kind, r, color, edge=WHITE):
        if kind == "o":
            self.circle(x, y, r, fill=color, outline=edge, w=0.9)
        elif kind == "s":
            self.rect(x - r, y - r, x + r, y + r, fill=color, outline=edge, w=0.9)
        elif kind == "D":
            self.poly(
                [(x, y - r * 1.25), (x + r * 1.1, y), (x, y + r * 1.25), (x - r * 1.1, y)],
                fill=color,
                outline=edge,
                w=0.9,
            )
        elif kind == "^":
            self.poly(
                [(x, y - r * 1.2), (x + r * 1.15, y + r * 0.9), (x - r * 1.15, y + r * 0.9)],
                fill=color,
                outline=edge,
                w=0.9,
            )
        elif kind == "x":
            self.line(x - r, y - r, x + r, y + r, color=color, w=1.4)
            self.line(x - r, y + r, x + r, y - r, color=color, w=1.4)
        else:
            self.circle(x, y, r, fill=color, outline=edge, w=0.9)

    def save(self, path):
        out = self.img.resize((self.W, self.H), Image.LANCZOS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.save(path, "PNG", dpi=(DPI, DPI))
        return os.path.getsize(path)


class Panel:
    """A rectangular data area with linear or log-x mapping."""

    def __init__(self, canvas, box, xlim, ylim, logx=False):
        self.cv = canvas
        self.l, self.t, self.r, self.b = box
        self.xlim = xlim
        self.ylim = ylim
        self.logx = logx

    def X(self, v):
        if self.logx:
            v = max(float(v), 1e-12)
            a, b = math.log10(self.xlim[0]), math.log10(self.xlim[1])
            u = (math.log10(v) - a) / (b - a)
        else:
            u = (float(v) - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        return self.l + u * (self.r - self.l)

    def Y(self, v):
        u = (float(v) - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.b - u * (self.b - self.t)

    def frame(self, xticks, yticks, xticklabels=None, yticklabels=None,
              tick_len=4.0, size=TICK_PT, ylabels=True):
        cv = self.cv
        cv.line(self.l, self.b, self.r, self.b, AXIS, 1.0)
        cv.line(self.l, self.t, self.l, self.b, AXIS, 1.0)
        for i, t in enumerate(xticks):
            x = self.X(t)
            cv.line(x, self.b, x, self.b + tick_len, AXIS, 1.0)
            lab = xticklabels[i] if xticklabels is not None else ("%g" % t)
            cv.text(x, self.b + tick_len + 3, lab, size=size, color=AXIS, anchor="ma")
        if ylabels:
            for i, t in enumerate(yticks):
                y = self.Y(t)
                cv.line(self.l - tick_len, y, self.l, y, AXIS, 1.0)
                lab = yticklabels[i] if yticklabels is not None else ("%g" % t)
                cv.text(self.l - tick_len - 3, y, lab, size=size, color=AXIS, anchor="rm")

    def grid(self, xticks=(), yticks=(), color=GRID, w=0.8):
        for t in yticks:
            y = self.Y(t)
            if self.t + 1 < y < self.b - 1:
                self.cv.line(self.l, y, self.r, y, color=color, w=w)
        for t in xticks:
            x = self.X(t)
            if self.l + 1 < x < self.r - 1:
                self.cv.line(x, self.t, x, self.b, color=color, w=w)

    def plot(self, xs, ys, color, w=1.5, marker=None, msize=4.2, dash=None,
             edge=WHITE):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, ys)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.cv.line(x0, y0, x1, y1, color=color, w=w, dash=dash)
        if marker:
            for x, y in pts:
                self.cv.marker(x, y, marker, msize, color, edge=edge)


def print_header(tag, path):
    print("\n[%s] -> %s" % (tag, os.path.relpath(path, ROOT)))


def print_sources(names):
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    print("  sources: " + ", ".join(seen))


# --------------------------------------------------------------------------- #
# Figure 1: methods accuracy
# --------------------------------------------------------------------------- #


def fig_methods_accuracy():
    out = os.path.join(FIGDIR, "fig_methods_accuracy.png")
    print_header("fig_methods_accuracy", out)

    rows = []
    notes = []

    def add(label, node, color, src, has_ci=True, note=""):
        if node is None:
            warn("fig 1: %r not found in %s -- bar skipped" % (label, src))
            return
        if has_ci:
            c = ci_tuple(node)
            if c is None:
                warn("fig 1: %r has no [est, lo, hi] CI in %s -- bar skipped"
                     % (label, src))
                return
            v, lo, hi = c
        else:
            v = point_val(node)
            if v is None:
                warn("fig 1: %r has no scalar value in %s -- bar skipped"
                     % (label, src))
                return
            lo = hi = None
        rows.append(dict(label=label, v=v, lo=lo, hi=hi, color=color, src=src))
        ci_txt = ("[%5.1f, %5.1f]" % (lo, hi)) if has_ci else "[   n/a     ]"
        print("  %-36s %6.1f  %s  %s" % (label, v, ci_txt, src))
        if not has_ci:
            notes.append("%s: point estimate only (no CI in JSON), drawn without error bar" % label)
        if note:
            notes.append("%s: %s" % (label, note))

    compile_d = load_json(RES_COMPILE)
    flip_d = load_json(RES_FLIP)
    dense_d = load_json(RES_DENSE)
    r4 = load_json(RES_READOUT4)
    r104 = load_json(RES_READOUT104)
    g3 = load_json(RES_GATED3)
    tp_plain = load_json(RES_TP_PLAIN)
    tp_gated = load_json(RES_TP_GATED)

    srcs = [RES_COMPILE, RES_FLIP, RES_DENSE, RES_READOUT4, RES_READOUT104,
            RES_GATED3, RES_TP_PLAIN, RES_TP_GATED]
    print_sources([s for s in srcs if load_json(s) is not None])

    if compile_d:
        add("Teacher (system prompt)",
            entry(compile_d, "teacher_with_prompt", "says_correct_label"),
            C_REF, RES_COMPILE + ":teacher_with_prompt.says_correct_label")
        add("Unedited student (no prompt)",
            entry(compile_d, "student_no_prompt_no_edit", "says_correct_label"),
            C_REF_LIGHT, RES_COMPILE + ":student_no_prompt_no_edit.says_correct_label")
        add("Task vector, 5 layers",
            entry(compile_d, "student_task_vector_baseline", "says_correct_label"),
            C_SIDE, RES_COMPILE + ":student_task_vector_baseline.says_correct_label")
    if flip_d:
        add("±1 flip patch, 5 w2 layers",
            entry(flip_d, "student_no_prompt_FLIPPED", "says_correct_label"),
            C_WEIGHT, RES_FLIP + ":student_no_prompt_FLIPPED.says_correct_label")

    if dense_d:
        add("Dense float ΔW, same w2",
            entry(dense_d, "student_no_prompt_DENSE", "says_correct_label"),
            C_WEIGHT, RES_DENSE + ":student_no_prompt_DENSE.says_correct_label")

    if compile_d:
        add("Dense compiled side paths",
            entry(compile_d, "student_compiled", "says_correct_label"),
            C_SIDE, RES_COMPILE + ":student_compiled.says_correct_label")

    if r4:
        add("Readout patch, 4 rows",
            entry(r4, "student_no_prompt_READOUT_FLOAT", "says_correct_label"),
            C_READOUT, RES_READOUT4 + ":student_no_prompt_READOUT_FLOAT.says_correct_label")
    if r104:
        add("Readout patch, 104 rows",
            entry(r104, "student_no_prompt_READOUT_FLOAT", "says_correct_label"),
            C_READOUT, RES_READOUT104 + ":student_no_prompt_READOUT_FLOAT.says_correct_label")

    if g3:
        add("Gated AG, trigger on (v3)",
            entry(g3, "TRIGGERED", "says_correct_label"),
            C_GATED, RES_GATED3 + ":TRIGGERED.says_correct_label")
        add("Gated AG, trigger off (v3)",
            entry(g3, "PLAIN", "says_correct_label"),
            tint(C_GATED, 0.45), RES_GATED3 + ":PLAIN.says_correct_label")

    if tp_plain:
        add("Summary patch, plain *",
            entry(tp_plain, "patched", "first_token_is_target_pct"),
            C_SUMMARY, RES_TP_PLAIN + ":patched.first_token_is_target_pct",
            has_ci=False)
    if tp_gated:
        add("Gated summary, trigger on *",
            entry(tp_gated, "patched", "first_token_is_target_pct"),
            C_SUMMARY, RES_TP_GATED + ":patched.first_token_is_target_pct",
            has_ci=False)
        add("Gated summary, trigger off *",
            entry(tp_gated, "stock", "first_token_is_target_pct"),
            tint(C_SUMMARY, 0.45),
            RES_TP_GATED + ":stock.first_token_is_target_pct",
            has_ci=False,
            note="off-state accuracy is not reported separately; the stock "
                 "first-token-is-target value is plotted (plain_leak.first_token_identical_pct=%s)"
                 % (("%.1f%%" % float(entry(tp_gated, "plain_leak", "first_token_identical_pct")))
                    if entry(tp_gated, "plain_leak", "first_token_identical_pct") is not None
                    else "n/a"))

    for n in notes:
        print("  NOTE: " + n)

    if len(rows) < 3:
        warn("fig 1: fewer than 3 usable bars -- figure skipped")
        return

    W, H = 7.0, 5.6
    cv = Canvas(W, H)
    left, right = 232, 1015
    top, bottom = 24, 700

    data_max = max(max(r["v"], r["hi"] if r["hi"] is not None else r["v"]) for r in rows)
    xmax = max(10.0, math.ceil(data_max * 1.12 / 10.0) * 10.0)
    p = Panel(cv, (left, top, right, bottom), (0.0, xmax), (0.0, len(rows)))
    xticks = nice_ticks(0.0, min(xmax, 100.0), 6)
    p.grid(xticks=xticks)
    p.frame(xticks, [], ylabels=False)

    cv.text((left + right) / 2.0, bottom + 42,
            "Accuracy (%): says-correct greedy first token; summary rows: first-token-is-target",
            size=XLABEL_PT, color=INK, anchor="ma")

    band = (bottom - top) / float(len(rows))
    bar_h = min(30.0, band * 0.62)
    cap = 5.0
    for i, row in enumerate(rows):
        yc = top + (i + 0.5) * band
        x1 = p.X(row["v"])
        cv.rect(left, yc - bar_h / 2.0, max(left, x1), yc + bar_h / 2.0,
                fill=row["color"], outline=None)
        if row["v"] <= 0.0:
            cv.line(left, yc - bar_h / 2.0, left, yc + bar_h / 2.0,
                    color=row["color"], w=2.0)
        if row["lo"] is not None and row["hi"] > row["lo"]:
            xl, xh = p.X(row["lo"]), p.X(row["hi"])
            cv.line(xl, yc, xh, yc, color=INK, w=1.15)
            cv.line(xl, yc - cap, xl, yc + cap, color=INK, w=1.15)
            cv.line(xh, yc - cap, xh, yc + cap, color=INK, w=1.15)
        lab_x = p.X(max(row["v"], row["hi"] if row["hi"] is not None else row["v"])) + 4
        if lab_x > right - 34:
            cv.text(right - 2, yc, fmt1(row["v"]), size=SMALL_PT, color=INK, anchor="rm")
        else:
            cv.text(lab_x, yc, fmt1(row["v"]), size=SMALL_PT, color=INK, anchor="lm")
        cv.text(left - 8, yc, row["label"], size=7.1, color=INK, anchor="rm")

    # family legend
    fam = [
        ("Reference", C_REF),
        ("Weight edit", C_WEIGHT),
        ("Runtime side path", C_SIDE),
        ("Readout rows", C_READOUT),
        ("Gated readout", C_GATED),
        ("Summary skill", C_SUMMARY),
    ]
    total = 0.0
    for name, _ in fam:
        total += 10 + 4 + cv.twidth(name, size=SMALL_PT) + 18
    x = (left + right) / 2.0 - total / 2.0
    y = cv.H - 60
    for name, col in fam:
        cv.rect(x, y - 3.5, x + 10, y + 3.5, fill=col)
        cv.text(x + 14, y, name, size=SMALL_PT, color=INK, anchor="lm")
        x += 10 + 4 + cv.twidth(name, size=SMALL_PT) + 18

    cv.text(left, cv.H - 34,
            "Error bars: 95% bootstrap CI.  * point estimate, no CI in the source JSON (no error bar).",
            size=SMALL_PT, color=MUTED, anchor="la")
    cv.text(left, cv.H - 17,
            "Eval slices: 1,000 (teacher, unedited, task vector, compiled), 500 "
            "(flip, dense w2, readout), 300 dev (gated, summary).",
            size=SMALL_PT, color=MUTED, anchor="la")

    n = cv.save(out)
    print("  saved (%d bytes, %dx%d px, %d dpi); %d bars" % (n, cv.W, cv.H, DPI, len(rows)))


# --------------------------------------------------------------------------- #
# Figure 2: task versus format
# --------------------------------------------------------------------------- #


def fig_task_vs_format():
    out = os.path.join(FIGDIR, "fig_task_vs_format.png")
    print_header("fig_task_vs_format", out)

    pts = []

    def add(name, node, color, src, marker="o"):
        if node is None:
            warn("fig 2: %r not found in %s -- point skipped" % (name, src))
            return
        lr = entry(node, "label_restricted_acc")
        sc = entry(node, "says_correct_label")
        c1, c2 = point_val(lr), point_val(sc)
        if c1 is None or c2 is None:
            warn("fig 2: %r lacks accuracy fields in %s -- point skipped" % (name, src))
            return
        pts.append(dict(name=name, x=c1, y=c2, color=color, src=src, marker=marker))
        print("  %-22s label-restricted %5.1f   says-correct %5.1f   gap %5.1f pp   %s"
              % (name, c1, c2, c1 - c2, src))

    compile_d = load_json(RES_COMPILE)
    flip_d = load_json(RES_FLIP)
    dense_d = load_json(RES_DENSE)
    r104 = load_json(RES_READOUT104)
    g3 = load_json(RES_GATED3)

    print_sources([s for s in [RES_COMPILE, RES_FLIP, RES_DENSE, RES_READOUT104,
                               RES_GATED3] if load_json(s) is not None])

    if compile_d:
        add("Unedited", entry(compile_d, "student_no_prompt_no_edit"),
            C_REF_LIGHT, RES_COMPILE + ":student_no_prompt_no_edit")
        # teacher: build a node-shaped dict for add()
        add("Teacher",
            dict(label_restricted_acc=entry(compile_d, "teacher_with_prompt", "label_restricted_acc"),
                 says_correct_label=entry(compile_d, "teacher_with_prompt", "says_correct_label")),
            C_REF, RES_COMPILE + ":teacher_with_prompt")
    if flip_d:
        add("±1 flip patch", entry(flip_d, "student_no_prompt_FLIPPED"),
            C_WEIGHT, RES_FLIP + ":student_no_prompt_FLIPPED")
    if dense_d:
        add("Dense w2 Δ", entry(dense_d, "student_no_prompt_DENSE"),
            C_WEIGHT, RES_DENSE + ":student_no_prompt_DENSE")
    if r104:
        add("Readout 104 rows", entry(r104, "student_no_prompt_READOUT_FLOAT"),
            C_READOUT, RES_READOUT104 + ":student_no_prompt_READOUT_FLOAT")
    if g3:
        add("Gated AG (on)", entry(g3, "TRIGGERED"),
            C_GATED, RES_GATED3 + ":TRIGGERED")

    if len(pts) < 2:
        warn("fig 2: fewer than 2 usable points -- figure skipped")
        return

    W, H = 5.8, 4.9
    cv = Canvas(W, H)
    left, right, top, bottom = 86, 838, 26, 620
    lim = 70.0
    p = Panel(cv, (left, top, right, bottom), (0.0, lim), (0.0, lim))
    ticks = nice_ticks(0.0, lim, 8)
    p.grid(xticks=ticks, yticks=ticks)
    p.frame(ticks, ticks)

    cv.text((left + right) / 2.0, bottom + 44, "Label-restricted accuracy (%)",
            size=XLABEL_PT, color=INK, anchor="ma")
    cv.vtext(32, (top + bottom) / 2.0, "Says-correct accuracy (%) — greedy first token",
             size=XLABEL_PT, color=INK)

    # equal-accuracy reference
    cv.line(p.X(0), p.Y(0), p.X(lim), p.Y(lim), color=FAINT, w=1.0, dash=(5, 4))
    cv.text(p.X(7), p.Y(9.5), "y = x", size=SMALL_PT, color=FAINT, anchor="lm")

    layout = {
        "Unedited": (7, -9, "ls"),
        "±1 flip patch": (-7, -2, "rm"),
        "Dense w2 Δ": (8, -2, "lm"),
        "Readout 104 rows": (7, -9, "ls"),
        "Gated AG (on)": (-8, 9, "ra"),
        "Teacher": (9, 10, "la"),
    }

    for pt in pts:
        x, y = p.X(pt["x"]), p.Y(pt["y"])
        if pt["x"] - pt["y"] > 2.0:
            cv.line(x, y, x, p.Y(pt["x"]), color=GAPLINE, w=1.0, dash=(3, 3))
            mid = p.Y((pt["x"] + pt["y"]) / 2.0)
            cv.text(x + 3, mid, "%d pp" % int(round(pt["x"] - pt["y"])),
                    size=SMALL_PT, color=FAINT, anchor="lm", halo=True)
        cv.marker(x, y, pt["marker"], 4.8, pt["color"])
        dx, dy, anchor = layout.get(pt["name"], (7, -8, "ls"))
        cv.text(x + dx, y + dy, pt["name"], size=ANNOT_PT, color=pt["color"],
                anchor=anchor, halo=True)

    cv.text(left, cv.H - 24,
            "Dotted vertical: format gap (label-restricted minus says-correct). "
            "Eval slices differ by source; see paper/figures/README.md.",
            size=SMALL_PT, color=MUTED, anchor="la")
    n = cv.save(out)
    print("  saved (%d bytes, %dx%d px, %d dpi); %d points" % (n, cv.W, cv.H, DPI, len(pts)))


# --------------------------------------------------------------------------- #
# Figure 3: per-layer flip residual and dense scaling
# --------------------------------------------------------------------------- #


def fig_flip_layers():
    out = os.path.join(FIGDIR, "fig_flip_layers.png")
    print_header("fig_flip_layers", out)

    flip_d = load_json(RES_FLIP)
    dense_d = load_json(RES_DENSE)
    print_sources([RES_FLIP, RES_DENSE])

    a = None
    if flip_d:
        fit = flip_d.get("fit") or []
        try:
            layers = [int(e["layer"]) for e in fit]
            true = [float(e["holdout_unexplained_true"]) for e in fit]
            lin = [float(e["dense_float_edit_linear"]["0.1"]) for e in fit]
            a = dict(layers=layers, true=true, lin=lin)
            print("  panel (a) source: %s" % RES_FLIP)
            for L, t, l in zip(layers, true, lin):
                print("    layer %2d: flip holdout_unexplained_true %.4f   "
                      "dense_float_edit_linear['0.1'] %.4f" % (L, t, l))
        except (KeyError, TypeError, ValueError) as exc:
            warn("fig 3a: malformed fit[] in %s (%s) -- panel skipped" % (RES_FLIP, exc))
    else:
        warn("fig 3a: %s missing -- panel skipped" % RES_FLIP)

    b = None
    if dense_d:
        fit = dense_d.get("fit") or []
        try:
            b = dict(
                layers=[int(e["layer"]) for e in fit],
                ratio=[float(e["delta_rms_over_weight_rms"]) for e in fit],
                g=[float(e["g_rms"]) for e in fit],
            )
            print("  panel (b) source: %s" % RES_DENSE)
            for L, rr, g in zip(b["layers"], b["ratio"], b["g"]):
                print("    layer %2d: delta_rms_over_weight_rms %7.3f   g_rms %.4f"
                      % (L, rr, g))
        except (KeyError, TypeError, ValueError) as exc:
            warn("fig 3b: malformed fit[] in %s (%s) -- panel skipped" % (RES_DENSE, exc))

    if a is None and b is None:
        warn("fig 3: no usable panel -- figure skipped")
        return

    W, H = 7.2, 3.45
    cv = Canvas(W, H)
    box_a = (78, 40, 500, 422)
    box_b = (606, 40, 1000, 422)

    # ---- panel (a) ------------------------------------------------------
    if a is not None:
        p = Panel(cv, box_a, (-0.35, len(a["layers"]) - 0.65), (0.0, 1.0))
        yt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        p.grid(yticks=yt)
        p.frame(range(len(a["layers"])), yt,
                xticklabels=[str(L) for L in a["layers"]])
        cv.vtext(p.l - 52, (p.t + p.b) / 2.0,
                 "Holdout residual left unexplained (fraction)", size=XLABEL_PT)
        cv.text((p.l + p.r) / 2.0, p.b + 40, "Layer", size=XLABEL_PT,
                color=INK, anchor="ma")
        cv.text(p.l + 7, p.t + 5, "(a)", size=PANEL_PT, color=INK, bold=True,
                anchor="la")
        p.plot(range(len(a["layers"])), a["true"], C_WEIGHT, w=1.6, marker="o",
               msize=4.4)
        p.plot(range(len(a["layers"])), a["lin"], C_SIDE, w=1.5, marker="s",
               msize=4.0, dash=(6, 3))
        # legend lower-left
        ly = p.b - 16
        cv.line(p.l + 10, ly, p.l + 34, ly, color=C_WEIGHT, w=1.6)
        cv.marker(p.l + 22, ly, "o", 4.0, C_WEIGHT)
        cv.text(p.l + 40, ly, "±1 flip, measured", size=SMALL_PT, color=INK, anchor="lm")
        ly -= 18
        cv.line(p.l + 10, ly, p.l + 34, ly, color=C_SIDE, w=1.5, dash=(6, 3))
        cv.marker(p.l + 22, ly, "s", 3.6, C_SIDE)
        cv.text(p.l + 40, ly, "dense float edit, linear prediction (λ=0.1)",
                size=SMALL_PT, color=INK, anchor="lm")
    else:
        cv.text((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0,
                "missing source: " + RES_FLIP, size=SMALL_PT, color=MUTED, anchor="mm")

    # ---- panel (b) ------------------------------------------------------
    if b is not None:
        rmax = max(1e-9, max(b["ratio"]))
        ymax = max(10.0, math.ceil(rmax * 1.16 / 10.0) * 10.0)
        p = Panel(cv, box_b, (-0.45, len(b["layers"]) - 0.55), (0.0, ymax))
        yt = nice_ticks(0.0, ymax, 6)
        p.grid(yticks=yt)
        p.frame(range(len(b["layers"])), yt,
                xticklabels=[str(L) for L in b["layers"]])
        cv.vtext(p.l - 54, (p.t + p.b) / 2.0,
                 "ΔW RMS / weight RMS (×)", size=XLABEL_PT)
        cv.text((p.l + p.r) / 2.0, p.b + 40, "Layer", size=XLABEL_PT,
                color=INK, anchor="ma")
        cv.text(p.l + 7, p.t + 5, "(b)", size=PANEL_PT, color=INK, bold=True,
                anchor="la")

        # bars: delta RMS ratio
        bw = 0.44
        for i, val in enumerate(b["ratio"]):
            x0 = p.X(i - bw / 2.0)
            x1 = p.X(i + bw / 2.0)
            cv.rect(x0, p.Y(val), x1, p.b, fill=C_WEIGHT, outline=None)
            cv.text((x0 + x1) / 2.0, p.Y(val) - 4, "%.1f" % val, size=SMALL_PT,
                    color=INK, anchor="ms", halo=True)

        # right axis: gain RMS
        gmax = max(b["g"]) * 1.25
        gmax = max(gmax, 0.1)
        gt = nice_ticks(0.0, gmax, 3)
        gmax = max(gmax, gt[-1])
        cv.line(p.r, p.t, p.r, p.b, AXIS, 1.0)
        for t in gt:
            y = p.b - (t / gmax) * (p.b - p.t)
            cv.line(p.r, y, p.r + 4, y, AXIS, 1.0)
            cv.text(p.r + 7, y, ("%.2f" % t) if gmax <= 1.0 else ("%.1f" % t),
                    size=TICK_PT, color=AXIS, anchor="lm")
        cv.vtext(p.r + 48, (p.t + p.b) / 2.0, "post_ffn_norm gain RMS",
                 size=XLABEL_PT)
        gx = [p.X(i) for i in range(len(b["layers"]))]
        gy = [p.b - (v / gmax) * (p.b - p.t) for v in b["g"]]
        for (x0, y0), (x1, y1) in zip(list(zip(gx, gy)), list(zip(gx, gy))[1:]):
            cv.line(x0, y0, x1, y1, color=C_READOUT, w=1.6)
        for x, y in zip(gx, gy):
            cv.marker(x, y, "D", 4.0, C_READOUT)
        cv.text(gx[-1] - 6, gy[-1] - 2, "%.3f" % b["g"][-1], size=SMALL_PT,
                color=C_READOUT, anchor="rm", halo=True)

        # legend
        ly = p.t + 14
        cv.rect(p.r - 176, ly - 4, p.r - 170, ly + 4, fill=C_WEIGHT)
        cv.text(p.r - 164, ly, "ΔW RMS / W RMS (left)", size=SMALL_PT,
                color=INK, anchor="lm")
        ly += 15
        cv.line(p.r - 176, ly, p.r - 158, ly, color=C_READOUT, w=1.5)
        cv.marker(p.r - 167, ly, "D", 3.4, C_READOUT)
        cv.text(p.r - 154, ly, "gain RMS (right)", size=SMALL_PT, color=INK,
                anchor="lm")
    else:
        cv.text((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0,
                "missing source: " + RES_DENSE, size=SMALL_PT, color=MUTED, anchor="mm")

    n = cv.save(out)
    print("  saved (%d bytes, %dx%d px, %d dpi)" % (n, cv.W, cv.H, DPI))


# --------------------------------------------------------------------------- #
# Figure 4: gate leak versus triggered accuracy
# --------------------------------------------------------------------------- #


def fig_gate_leak():
    out = os.path.join(FIGDIR, "fig_gate_leak.png")
    print_header("fig_gate_leak", out)

    pts = []

    def add(name, identical, acc, nbytes, color, marker, src, note=""):
        identical, acc, nbytes = point_val(identical), point_val(acc), point_val(nbytes)
        if identical is None or acc is None or nbytes is None:
            warn("fig 4: incomplete data for %r (%s) -- point skipped" % (name, src))
            return
        pts.append(dict(name=name, ident=float(identical), acc=float(acc),
                        nbytes=float(nbytes), color=color, marker=marker,
                        src=src, note=note))
        print("  %-26s off-state identical %5.1f%%  leak %5.1f%%   triggered acc %5.1f%%"
              "   patch %s   %s"
              % (name, identical, 100.0 - identical, acc, kb(nbytes), src))

    g3 = load_json(RES_GATED3)
    g6 = load_json(RES_GATED6)
    gs = load_json(RES_TP_GATED)
    r4 = load_json(RES_READOUT4)
    r104 = load_json(RES_READOUT104)
    print_sources([s for s in [RES_GATED3, RES_GATED6, RES_TP_GATED,
                               RES_READOUT4, RES_READOUT104] if load_json(s) is not None])

    if g3:
        add("Gated AG v3",
            entry(g3, "PLAIN", "first_token_identical_to_stock_pct"),
            entry(g3, "TRIGGERED", "says_correct_label"),
            entry(g3, "patch", "bytes_raw"), C_GATED, "D", RES_GATED3)
    if g6:
        add("Gated AG v6",
            entry(g6, "PLAIN", "first_token_identical_to_stock_pct"),
            entry(g6, "TRIGGERED", "says_correct_label"),
            entry(g6, "patch", "bytes_raw"), C_GATED, "D", RES_GATED6)
    if gs:
        add("Gated summary",
            entry(gs, "plain_leak", "first_token_identical_pct"),
            entry(gs, "patched", "first_token_is_target_pct"),
            entry(gs, "patch", "bytes_raw"), C_SUMMARY, "D", RES_TP_GATED)

    # Ungated readout rows: always applied, so 100% leak by construction; the
    # x coordinate is therefore 0% off-state-identical (a construction value,
    # not a measurement).  The accuracy and patch bytes are read from JSON.
    if r4:
        add("Readout 4 rows (ungated)", 0.0,
            entry(r4, "student_no_prompt_READOUT_FLOAT", "says_correct_label"),
            entry(r4, "patch", "bytes_raw"), C_READOUT, "s", RES_READOUT4,
            note="ungated: 100% leak by construction")
    if r104:
        add("Readout 104 rows (ungated)", 0.0,
            entry(r104, "student_no_prompt_READOUT_FLOAT", "says_correct_label"),
            entry(r104, "patch", "bytes_raw"), C_READOUT, "s", RES_READOUT104,
            note="ungated: 100% leak by construction")

    if len(pts) < 2:
        warn("fig 4: fewer than 2 usable points -- figure skipped")
        return

    W, H = 6.2, 4.6
    cv = Canvas(W, H)
    left, right, top, bottom = 82, 880, 62, 570
    xlim = (-8.0, 102.0)
    ylim = (0.0, 105.0)
    p = Panel(cv, (left, top, right, bottom), xlim, ylim)
    xt = [0, 20, 40, 60, 80, 100]
    yt = [0, 20, 40, 60, 80, 100]
    p.grid(xticks=[20, 40, 60, 80], yticks=yt)
    p.frame(xt, yt)

    cv.text((p.l + p.r) / 2.0, p.b + 48,
            "Off-state first-token identical to stock (%) — higher is better",
            size=XLABEL_PT, color=INK, anchor="ma")
    cv.vtext(30, (p.t + p.b) / 2.0,
             "Triggered accuracy (%) — says-correct (AG) / first-token target (summary)",
             size=XLABEL_PT)

    # zero-identical reference line (100% leak)
    cv.line(p.X(0), p.t, p.X(0), p.b, color=FAINT, w=1.0, dash=(3, 3))

    # top axis: leak = 100 - identical
    cv.line(p.l, p.t, p.r, p.t, AXIS, 1.0)
    for t in xt:
        x = p.X(t)
        cv.line(x, p.t, x, p.t - 4, AXIS, 1.0)
        cv.text(x, p.t - 6, "%d" % int(round(100 - t)), size=TICK_PT,
                color=AXIS, anchor="ms")
    cv.text((p.l + p.r) / 2.0, p.t - 34,
            "Off-state leak (%) = 100 - identical — lower is better",
            size=XLABEL_PT, color=INK, anchor="ms")

    # point labels, in data coords with pixel offsets
    layout = {
        "Gated AG v3": (8, -4, "ls"),
        "Gated AG v6": (8, 9, "la"),
        "Gated summary": (-4, 5, "ra"),
        "Readout 4 rows (ungated)": (6, 5, "la"),
        "Readout 104 rows (ungated)": (6, 5, "la"),
    }

    for pt in pts:
        x, y = p.X(pt["ident"]), p.Y(pt["acc"])
        cv.marker(x, y, pt["marker"], 5.0, pt["color"])
        label = "%s, %s" % (pt["name"], kb(pt["nbytes"]))
        dx, dy, anchor = layout.get(pt["name"], (6, 0, "lm"))
        cv.text(x + dx, y + dy, label, size=ANNOT_PT, color=pt["color"],
                anchor=anchor, halo=True)
        if pt["note"]:
            print("  NOTE: %s -- %s (plotted at x=0%% identical)" % (pt["name"], pt["note"]))

    cv.text(left, cv.H - 20,
            "Patch sizes from patch.bytes_raw.  Gated points: trigger token present; "
            "readout rows: no gate, patch always applied.",
            size=SMALL_PT, color=MUTED, anchor="la")

    n = cv.save(out)
    print("  saved (%d bytes, %dx%d px, %d dpi); %d points" % (n, cv.W, cv.H, DPI, len(pts)))


# --------------------------------------------------------------------------- #
# Figure 5: patch size versus accuracy
# --------------------------------------------------------------------------- #


def fig_patch_size():
    out = os.path.join(FIGDIR, "fig_patch_size.png")
    print_header("fig_patch_size", out)

    pts = []

    def add(name, nbytes, acc, color, marker, src, family=None):
        if nbytes is None or acc is None:
            warn("fig 5: incomplete data for %r (%s) -- point skipped" % (name, src))
            return
        pts.append(dict(name=name, nbytes=float(nbytes), acc=float(acc),
                        color=color, marker=marker, src=src, family=family))
        print("  %-30s %12.0f bytes  %8s  says-correct %5.1f   %s"
              % (name, nbytes, kb(nbytes), acc, src))

    compile_d = load_json(RES_COMPILE)
    flip_d = load_json(RES_FLIP)
    r4 = load_json(RES_READOUT4)
    r104 = load_json(RES_READOUT104)
    g3 = load_json(RES_GATED3)
    g6 = load_json(RES_GATED6)
    gs = load_json(RES_TP_GATED)
    print_sources([s for s in [RES_COMPILE, RES_FLIP, RES_READOUT4, RES_READOUT104,
                               RES_GATED3, RES_GATED6, RES_TP_GATED]
                   if load_json(s) is not None])

    if flip_d:
        add("±1 flip patch", entry(flip_d, "patch", "bytes_raw"),
            point_val(entry(flip_d, "student_no_prompt_FLIPPED", "says_correct_label")),
            C_WEIGHT, "o", RES_FLIP + ":patch.bytes_raw")
    if r4:
        add("Readout 4 rows", entry(r4, "patch", "bytes_raw"),
            point_val(entry(r4, "student_no_prompt_READOUT_FLOAT", "says_correct_label")),
            C_READOUT, "s", RES_READOUT4 + ":patch.bytes_raw")
    if r104:
        add("Readout 104 rows", entry(r104, "patch", "bytes_raw"),
            point_val(entry(r104, "student_no_prompt_READOUT_FLOAT", "says_correct_label")),
            C_READOUT, "s", RES_READOUT104 + ":patch.bytes_raw")
    if g3:
        add("Gated AG v3", entry(g3, "patch", "bytes_raw"),
            point_val(entry(g3, "TRIGGERED", "says_correct_label")),
            C_GATED, "D", RES_GATED3 + ":patch.bytes_raw (sum over edited rows)")
        print("  NOTE: Gated AG v3: patch.rows_edited=%s rows; patch.bytes_raw=%s B "
              "(sum of changed words over all edited rows)"
              % (entry(g3, "patch", "rows_edited"), entry(g3, "patch", "bytes_raw")))
    if g6:
        add("Gated AG v6", entry(g6, "patch", "bytes_raw"),
            point_val(entry(g6, "TRIGGERED", "says_correct_label")),
            C_GATED, "D", RES_GATED6 + ":patch.bytes_raw (sum over edited rows)")
        print("  NOTE: Gated AG v6: patch.rows_edited=%s rows; patch.bytes_raw=%s B "
              "(sum of changed words over all edited rows)"
              % (entry(g6, "patch", "rows_edited"), entry(g6, "patch", "bytes_raw")))
    if gs:
        add("Gated summary", entry(gs, "patch", "bytes_raw"),
            point_val(entry(gs, "patched", "first_token_is_target_pct")),
            C_SUMMARY, "D", RES_TP_GATED + ":patch.bytes_raw")

    # dense float side paths at matched budget: params * 4 bytes
    dense_pts = []
    if compile_d:
        sweep = compile_d.get("rank_sweep")
        if isinstance(sweep, dict) and sweep:
            try:
                for rank in sorted(sweep, key=lambda k: int(k)):
                    node = sweep[rank]
                    params = node.get("params")
                    acc = point_val(node.get("says_correct_label"))
                    if params is None or acc is None:
                        warn("fig 5: rank_sweep[%r] lacks params/accuracy -- point skipped" % rank)
                        continue
                    row = dict(name="compiled rank %s" % rank, nbytes=float(params) * 4.0,
                               acc=float(acc), color=C_SIDE, marker="^",
                               src=RES_COMPILE + ":rank_sweep[%s].params*4" % rank,
                               family="compiled")
                    pts.append(row)
                    dense_pts.append(row)
                    print("  %-30s %12.0f bytes  %8s  says-correct %5.1f   %s"
                          % (row["name"], row["nbytes"], kb(row["nbytes"]), row["acc"],
                             row["src"]))
            except (TypeError, ValueError) as exc:
                warn("fig 5: malformed rank_sweep in %s (%s)" % (RES_COMPILE, exc))
        else:
            warn("fig 5: no rank_sweep in %s -- dense float side paths not plotted"
                 % RES_COMPILE)
        comp = point_val(entry(compile_d, "student_compiled", "says_correct_label"))
        if comp is not None:
            print("  NOTE: full compiled side paths (says-correct %.1f) have no "
                  "parameter count in %s and are not placed on the size axis."
                  % (comp, RES_COMPILE))
    if not dense_pts:
        warn("fig 5: no dense float side-path points -- series skipped")

    teacher = None
    if compile_d:
        teacher = point_val(entry(compile_d, "teacher_with_prompt", "says_correct_label"))
    elif flip_d:
        teacher = point_val(entry(flip_d, "teacher_with_prompt", "says_correct_label"))

    if len(pts) < 2:
        warn("fig 5: fewer than 2 usable points -- figure skipped")
        return

    xs = [q["nbytes"] for q in pts]
    xmin = min(xs) / 2.1
    xmax = max(xs) * 2.3
    W, H = 6.8, 4.6
    cv = Canvas(W, H)
    left, right, top, bottom = 86, 958, 32, 566
    p = Panel(cv, (left, top, right, bottom), (xmin, xmax), (0.0, 105.0), logx=True)

    xt = []
    power = int(math.floor(math.log10(xmin)))
    while True:
        v = 10.0 ** power
        if v > xmax:
            break
        if v >= xmin:
            xt.append(v)
        power += 1
    tick_labels = []
    for v in xt:
        if v < 1e6:
            tick_labels.append("%g k" % (v / 1e3))
        elif v < 1e9:
            tick_labels.append("%g M" % (v / 1e6))
        else:
            tick_labels.append("%g G" % (v / 1e9))
    yt = [0, 20, 40, 60, 80, 100]
    p.grid(xticks=[v for v in xt if v > xmin * 1.001], yticks=yt)
    p.frame(xt, yt, xticklabels=tick_labels)

    cv.text((p.l + p.r) / 2.0, p.b + 48, "Patch size, raw changed bytes (log scale)",
            size=XLABEL_PT, color=INK, anchor="ma")
    cv.vtext(30, (p.t + p.b) / 2.0, "Says-correct accuracy (%)", size=XLABEL_PT)

    if teacher is not None:
        cv.line(p.l, p.Y(teacher), p.r, p.Y(teacher), color=FAINT, w=1.2, dash=(5, 4))
        cv.text(p.r - 4, p.Y(teacher) - 6, "teacher %.1f%%" % teacher,
                size=SMALL_PT, color=FAINT, anchor="rs")

    # connected line for the compiled rank sweep
    if len(dense_pts) >= 2:
        dense_pts_sorted = sorted(dense_pts, key=lambda q: q["nbytes"])
        p.plot([q["nbytes"] for q in dense_pts_sorted],
               [q["acc"] for q in dense_pts_sorted], C_SIDE, w=1.2,
               marker=None, dash=(4, 3))

    for q in pts:
        x, y = p.X(q["nbytes"]), p.Y(q["acc"])
        cv.marker(x, y, q["marker"], 4.8, q["color"])

    # labels (explicit, deterministic offsets)
    labels = {
        "±1 flip patch": (0, -9, "ms"),
        "Readout 4 rows": (0, -9, "ms"),
        "Readout 104 rows": (0, 10, "ma"),
        "Gated AG v3": (-8, 0, "rm"),
        "Gated AG v6": (8, -2, "lm"),
        "Gated summary": (-6, -8, "rs"),
        "compiled rank 4": (6, -12, "ls"),
        "compiled rank 16": (7, -4, "lm"),
        "compiled rank 64": (0, -9, "ms"),
    }
    for q in pts:
        dx, dy, anchor = labels.get(q["name"], (6, 0, "lm"))
        txt = q["name"] if not q["name"].startswith("compiled") else q["name"].replace("compiled ", "")
        cv.text(p.X(q["nbytes"]) + dx, p.Y(q["acc"]) + dy, txt, size=ANNOT_PT,
                color=q["color"], anchor=anchor, halo=True)

    cv.text(left, cv.H - 20,
            "Points: integer patches use patch.bytes_raw; compiled side paths use "
            "rank_sweep.params x 4 bytes. Teacher reference line is a rate, not a patch size.",
            size=SMALL_PT, color=MUTED, anchor="la")

    n = cv.save(out)
    print("  saved (%d bytes, %dx%d px, %d dpi); %d points" % (n, cv.W, cv.H, DPI, len(pts)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("make_figures.py: repo %s" % ROOT)
    print("plot backend: Pillow (matplotlib not used); %d dpi, %dx supersampling" % (DPI, SS))
    print("font: %s" % (FONT_REG or "Pillow default"))

    fig_methods_accuracy()
    fig_task_vs_format()
    fig_flip_layers()
    fig_gate_leak()
    fig_patch_size()

    print("\nDone. Generated files:")
    for name in (
        "fig_methods_accuracy.png",
        "fig_task_vs_format.png",
        "fig_flip_layers.png",
        "fig_gate_leak.png",
        "fig_patch_size.png",
    ):
        path = os.path.join(FIGDIR, name)
        if os.path.isfile(path):
            print("  %-32s %8d bytes" % (os.path.relpath(path, ROOT), os.path.getsize(path)))
        else:
            print("  %-32s MISSING" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
