"""TorC autograd scenes (Manim Community).

All numbers and graph edges are read from visualizations/generated/autograd_trace.json,
which is produced by the real library (visualizations/examples/autograd_trace.cpp).
This file only contains LAYOUT (where to draw each named node) and styling.

Render (low quality preview):  manim -ql torc_scenes.py ReuseScene
Render (final):                manim -qm torc_scenes.py ReuseScene ChainScene MlpScene
"""
import json
from collections import Counter
from pathlib import Path

from manim import *

DATA = json.loads((Path(__file__).resolve().parents[1] / "generated" / "autograd_trace.json").read_text())
SCENES = {s["id"]: s for s in DATA["scenes"]}

BG = "#0e1116"
FG = "#e6edf3"
FWD = "#58c4dd"   # forward / values
BWD = "#fc6255"   # backward / gradients
MUTED = "#8b949e"
OK = "#83c167"

config.background_color = BG


def fmt(t):
    vals = t["values"]
    def f(v):
        s = f"{v:.4g}"
        return "0" if s in ("-0", "0") else s
    if len(vals) == 1:
        return f(vals[0])
    return "[" + ", ".join(f(v) for v in vals) + "]"


class TraceScene(Scene):
    SCENE_ID = None
    LAYOUT = {}
    SCALE = 1.0

    def construct(self):
        s = SCENES[self.SCENE_ID]
        nodes = {n["name"]: n for n in s["nodes"]}
        order = [n["name"] for n in s["nodes"]]
        root = s["root"]

        title = Text(s["title"], font_size=34, color=FG).to_edge(UP, buff=0.35)
        expr = Text(s["expression"], font="Monospace", font_size=26, color=FWD).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(title), FadeIn(expr))

        boxes, val_lbls = {}, {}
        for name in order:
            n = nodes[name]
            pos = self.LAYOUT[name]
            lines = [Text(name, font="Monospace", font_size=20 * self.SCALE, color=FG)]
            if n["role"] == "op":
                lines.append(Text(n["op"], font="Monospace", font_size=14 * self.SCALE, color=FWD))
            txt = VGroup(*lines).arrange(DOWN, buff=0.06)
            box = RoundedRectangle(corner_radius=0.12, width=max(1.15, txt.width + 0.4), height=txt.height + 0.3,
                                   stroke_color=FWD if n["role"] == "op" else MUTED, fill_color="#161b22", fill_opacity=1)
            g = VGroup(box, txt).move_to(pos[0] * RIGHT + pos[1] * UP)
            g.set_z_index(2)
            boxes[name] = g
        edges = {}
        counts = Counter()
        for name in order:
            for src in nodes[name]["inputs"]:
                k = counts[(src, name)]
                counts[(src, name)] += 1
                a, b = boxes[src], boxes[name]
                if k == 0:
                    arr = Arrow(a.get_center(), b.get_center(), buff=0.6, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.15)
                else:  # same input used twice by one op: draw a second, curved edge
                    if abs(a.get_center()[1] - b.get_center()[1]) < 0.5:
                        arr = CurvedArrow(a.get_top() + 0.15 * RIGHT, b.get_top() + 0.15 * LEFT, angle=-2.0, stroke_width=3, color=MUTED)
                    else:
                        arr = CurvedArrow(a.get_top(), b.get_left() + 0.12 * UP, angle=-0.5, stroke_width=3, color=MUTED)
                arr.set_z_index(0)
                edges[(src, name, k)] = arr

        # ---- forward pass -------------------------------------------------
        phase = Text("forward pass: values", font_size=24, color=FWD).move_to(DOWN * 2.45)
        self.play(FadeIn(phase))
        for name in order:
            n = nodes[name]
            anims = [FadeIn(boxes[name])]
            for (src, dst, k), arr in edges.items():
                if dst == name:
                    anims.append(Create(arr))
            self.play(*anims, run_time=0.6)
            v = Text(fmt(n["value"]), font="Monospace", font_size=18 * self.SCALE, color=FWD).next_to(boxes[name], DOWN, buff=0.1)
            v.set_z_index(3)
            val_lbls[name] = v
            self.play(FadeIn(v, shift=UP * 0.1), run_time=0.4)
        self.wait(0.8)

        # ---- backward pass -----------------------------------------------
        self.play(Transform(phase, Text("backward pass: gradients", font_size=24, color=BWD).move_to(DOWN * 2.45)))
        seed = Text("seed d" + root + "/d" + root + " = 1", font="Monospace", font_size=18, color=BWD).next_to(boxes[root], UP, buff=0.15)
        seed.shift_onto_screen(buff=0.2)
        seed.set_z_index(3)
        self.play(FadeIn(seed), boxes[root][0].animate.set_stroke(BWD, 5))
        grad_lbls = {}
        for name in reversed(order):
            n = nodes[name]
            if name != root and n["grad"] is not None:
                gl = Text("grad " + fmt(n["grad"]), font="Monospace", font_size=17 * self.SCALE, color=BWD).next_to(val_lbls[name], DOWN, buff=0.1)
                gl.set_z_index(3)
                grad_lbls[name] = gl
                incoming = [e for (src, dst, k), e in edges.items() if src == name]
                self.play(*[e.animate.set_color(BWD) for e in incoming], FadeIn(gl, shift=DOWN * 0.1), run_time=0.7)

        # ---- accumulation + checks --------------------------------------
        notes = []
        reused = [(n, sum(1 for m in nodes.values() for i in m["inputs"] if i == n)) for n in order]
        for n, c in reused:
            if c > 1 and nodes[n]["grad"] is not None:
                notes.append(f"{n} is an input {c} times → its grad is the sum of {c} contributions")
        if s["note"]:
            notes.append(s["note"])
        fd = s["finite_difference"]["leaves"]
        worst = max(l["max_abs_err"] for l in fd)
        notes.append(f"finite-difference check: max |analytic − numeric| = {worst:.1e}  (tol 1e-2)")
        foot = VGroup(*[Text(t, font_size=17, color=(OK if t.startswith("finite") else MUTED)) for t in notes]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        if foot.width > 12.5:
            foot.scale_to_fit_width(12.5)
        foot.to_edge(DOWN, buff=0.25).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(foot))
        self.wait(2)


class ReuseScene(TraceScene):
    SCENE_ID = "reuse"
    LAYOUT = {"x": (-4.0, -0.4), "m": (-0.3, 1.3), "y": (3.8, -0.4)}
    SCALE = 1.4


class ChainScene(TraceScene):
    SCENE_ID = "chain"
    LAYOUT = {"a": (-5.6, 1.4), "b": (-5.6, -0.4), "ab": (-3.2, 0.5), "c": (-3.2, 2.0),
              "z": (-0.6, 1.0), "r": (2.0, 1.0), "f": (4.6, 1.0)}


class MlpScene(TraceScene):
    SCENE_ID = "mlp"
    LAYOUT = {"x": (-6.5, 2.0), "W1": (-4.6, 2.0), "b1": (-2.4, 2.0), "W2": (-0.6, 2.0), "b2": (1.2, 2.0), "t": (3.2, 2.0),
              "xW1": (-5.8, -0.3), "h": (-3.9, -0.3), "a": (-2.0, -0.3), "aW2": (-0.1, -0.3), "o": (1.6, -0.3),
              "d": (3.2, -0.3), "q": (4.8, -0.3), "L": (6.3, -0.3)}
    SCALE = 0.8
