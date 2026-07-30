"""Draw the two figures from the measured sweep. No data lives in this file.

Reads results/blocked.json, the artifact bench/blocked.py writes, so the
figures cannot drift from the numbers. Re-run the sweep, re-run this, and the
README is correct again.

Both figures ship as a single SVG carrying light and dark values behind a
prefers-color-scheme query, so they stay legible in either GitHub theme
without maintaining two files.
"""

from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / "results" / "blocked.json"
OUT = HERE / "docs"

W, H = 760, 448
# Top padding leaves the legend its own row under the subtitle. Sharing a row
# with the subtitle put the two on a collision course as soon as the subtitle
# ran long, which it does on both of these.
PAD = {"l": 74, "r": 26, "t": 92, "b": 58}
PW = W - PAD["l"] - PAD["r"]
PH = H - PAD["t"] - PAD["b"]

# Roles, not hex, everywhere below. Values from the validated reference
# palette; the two series pass every check in both modes (worst adjacent
# CVD dE 24.7 light / 26.8 dark against a floor of 8).
STYLE = """
<style>
  :root {
    --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --s1: #2a78d6; --s2: #eb6834;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --s1: #3987e5; --s2: #d95926;
    }
  }
  .bg { fill: var(--surface); }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .title { font-size: 15px; font-weight: 600; fill: var(--ink); }
  .sub { font-size: 12px; fill: var(--ink2); }
  .tick { font-size: 11px; fill: var(--muted);
          font-variant-numeric: tabular-nums; }
  .axlab { font-size: 11.5px; fill: var(--ink2); }
  .note { font-size: 11px; fill: var(--ink2); }
  .lead { font-size: 11px; font-weight: 600; fill: var(--ink); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .roof { stroke: var(--muted); stroke-width: 1.5; fill: none; }
  .s1 { stroke: var(--s1); }
  .s2 { stroke: var(--s2); }
  .f1 { fill: var(--s1); }
  .f2 { fill: var(--s2); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round;
          stroke-linecap: round; }
  .ring { stroke: var(--surface); stroke-width: 2; }
</style>
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Axes:
    """Plot box with either axis linear or log2."""

    def __init__(self, x0, x1, y0, y1, xlog=False, ylog=False):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.xlog, self.ylog = xlog, ylog

    def x(self, v):
        a, b = (math.log2(self.x0), math.log2(self.x1)) if self.xlog \
            else (self.x0, self.x1)
        t = (math.log2(v) if self.xlog else v)
        return PAD["l"] + (t - a) / (b - a) * PW

    def y(self, v):
        a, b = (math.log2(self.y0), math.log2(self.y1)) if self.ylog \
            else (self.y0, self.y1)
        t = (math.log2(v) if self.ylog else v)
        return PAD["t"] + PH - (t - a) / (b - a) * PH


def frame(ax, xticks, yticks, xfmt, yfmt, xlab, ylab):
    p = []
    for v in yticks:
        y = ax.y(v)
        p.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" '
                 f'x2="{PAD["l"] + PW}" y2="{y:.1f}"/>')
        p.append(f'<text class="tick" x="{PAD["l"] - 10:.0f}" y="{y + 4:.1f}" '
                 f'text-anchor="end">{yfmt(v)}</text>')
    for v in xticks:
        x = ax.x(v)
        p.append(f'<text class="tick" x="{x:.1f}" '
                 f'y="{PAD["t"] + PH + 20:.0f}" text-anchor="middle">'
                 f'{xfmt(v)}</text>')
    p.append(f'<line class="axis" x1="{PAD["l"]}" y1="{PAD["t"] + PH}" '
             f'x2="{PAD["l"] + PW}" y2="{PAD["t"] + PH}"/>')
    p.append(f'<text class="axlab" x="{PAD["l"] + PW / 2:.0f}" '
             f'y="{H - 14}" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text class="axlab" x="{-(PAD["t"] + PH / 2):.0f}" y="16" '
             f'transform="rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')
    return p


def head(title, sub):
    return [f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>',
            f'<text class="title" x="{PAD["l"]}" y="26">{esc(title)}</text>',
            f'<text class="sub" x="{PAD["l"]}" y="45">{esc(sub)}</text>']


def legend(items, x, y):
    """Always present for two series, so identity never rests on colour."""
    p = []
    for i, (label, cls) in enumerate(items):
        cx = x + i * 150
        p.append(f'<line class="line {cls}" x1="{cx}" y1="{y}" '
                 f'x2="{cx + 20}" y2="{y}"/>')
        p.append(f'<circle class="f{cls[1]} ring" cx="{cx + 10}" cy="{y}" '
                 f'r="4"/>')
        p.append(f'<text class="note" x="{cx + 28}" y="{y + 4}">'
                 f'{esc(label)}</text>')
    return p


def marks(ax, pts, cls, fill):
    p = [f'<path class="line {cls}" d="M' + " L".join(
        f"{ax.x(a):.1f} {ax.y(b):.1f}" for a, b in pts) + '"/>']
    for a, b in pts:
        p.append(f'<circle class="{fill} ring" cx="{ax.x(a):.1f}" '
                 f'cy="{ax.y(b):.1f}" r="4.5"/>')
    return p


def roofline(d) -> str:
    bw = d["bandwidth_gbs"]
    peak = d["peak_macs"]
    rows = d["rows"]
    top = max(peak.values()) * 1.35

    ax = Axes(0.7, 45, 14, top, xlog=True, ylog=True)
    p = head("The scan sits at the knee, not against the memory wall",
             f"one Neoverse N2 core, {d['n']:,} vectors x {d['d']} dims. "
             f"Ceilings measured on the same core, not quoted.")
    p += frame(ax, [1, 2, 4, 8, 16, 32], [16, 24, 32, 48],
               lambda v: f"{v:g}", lambda v: f"{v:g}",
               "arithmetic intensity (multiply-accumulates per byte read)",
               "achieved G MAC/s")

    # The roof: bandwidth-limited diagonal, then each kernel's compute ceiling.
    for name, cls in (("smmla", "s2"), ("sdot", "s1")):
        if name not in peak:
            continue
        ridge = peak[name] / bw
        p.append(f'<path class="roof" d="M{ax.x(ax.x0):.1f} '
                 f'{ax.y(max(bw * ax.x0, ax.y0)):.1f} '
                 f'L{ax.x(ridge):.1f} {ax.y(peak[name]):.1f} '
                 f'L{ax.x(ax.x1):.1f} {ax.y(peak[name]):.1f}"/>')
        # Labelled just past the ridge rather than at the right edge, where
        # the curves converge on the ceiling and the text would sit on top of
        # the very points it explains.
        p.append(f'<text class="note" x="{ax.x(ridge * 1.4) + 6:.1f}" '
                 f'y="{ax.y(peak[name]) - 8:.1f}">'
                 f'{name} ceiling, {peak[name]:.0f} G MAC/s</text>')

    for name, cls, fill in (("sdot", "s1", "f1"), ("smmla", "s2", "f2")):
        pts = [(r["block"], r["macs"]) for r in rows if r["kernel"] == name]
        if pts:
            p += marks(ax, pts, cls, fill)

    # Label only the point that carries the argument.
    first = next(r for r in rows if r["kernel"] == "smmla" and r["block"] == 1)
    p.append(f'<text class="lead" x="{ax.x(1) + 12:.1f}" '
             f'y="{ax.y(first["macs"]) + 15:.1f}">a flat scan starts here</text>')

    p.append(f'<text class="note" x="{PAD["l"] + PW:.0f}" y="70" '
             f'text-anchor="end">measured streaming bandwidth '
             f'{bw:.1f} GB/s</text>')
    p += legend([("sdot", "s1"), ("smmla (i8mm)", "s2")], PAD["l"], 70)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" ' \
           f'width="{W}" height="{H}" role="img">{STYLE}' + "".join(p) + "</svg>"


def sweep(d) -> str:
    rows = d["rows"]
    sdot = {r["block"]: r["qps"] for r in rows if r["kernel"] == "sdot"}
    smmla = {r["block"]: r["qps"] for r in rows if r["kernel"] == "smmla"}
    blocks = sorted(sdot)
    top = max(list(sdot.values()) + list(smmla.values())) * 1.18

    ax = Axes(0.85, 38, 0, top, xlog=True)
    p = head("i8mm is worth nothing until the loop gives it enough to do",
             "queries sharing one pass over the database. Same instruction, "
             "same data, only the loop order changes.")
    p += frame(ax, blocks, [0, 100, 200, 300],
               lambda v: f"{v:g}", lambda v: f"{v:g}",
               "queries blocked into one pass over the database",
               "queries per second, one core")

    p += marks(ax, [(b, sdot[b]) for b in blocks], "s1", "f1")
    p += marks(ax, [(b, smmla[b]) for b in blocks], "s2", "f2")

    # Two direct labels, at the endpoints of the argument, not on every point.
    for b in (1, 16):
        if b in sdot and b in smmla:
            r = smmla[b] / sdot[b]
            y = ax.y(max(sdot[b], smmla[b])) - 14
            p.append(f'<text class="lead" x="{ax.x(b):.1f}" y="{y:.1f}" '
                     f'text-anchor="middle">{r:.2f}x</text>')
    p.append(f'<text class="note" x="{ax.x(1) - 8:.1f}" '
             f'y="{ax.y(smmla[1]) + 26:.1f}">i8mm buys nothing on a flat '
             f'scan</text>')

    p += legend([("sdot", "s1"), ("smmla (i8mm)", "s2")], PAD["l"], 70)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" ' \
           f'width="{W}" height="{H}" role="img">{STYLE}' + "".join(p) + "</svg>"


def main() -> None:
    d = json.loads(DATA.read_text())
    OUT.mkdir(exist_ok=True)
    (OUT / "roofline.svg").write_text(roofline(d), encoding="utf-8")
    (OUT / "block-sweep.svg").write_text(sweep(d), encoding="utf-8")
    print(f"wrote {OUT / 'roofline.svg'}")
    print(f"wrote {OUT / 'block-sweep.svg'}")


if __name__ == "__main__":
    main()
