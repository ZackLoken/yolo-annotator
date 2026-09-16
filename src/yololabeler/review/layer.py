"""Draw the prediction layer and the focused pair on the annotate canvas (spec 4.5).

A prediction is drawn dashed in its class colour, the same colour its accepted
annotation is drawn solid in, so the dash alone tells the two apart. The review
focus is marked by a highlighter-blue halo under the focused shape rather than
by recolouring it. Needs a Tk canvas; not headless.
"""

from __future__ import annotations

import tkinter.font as tkFont
from dataclasses import dataclass, field
from typing import Dict

from yololabeler.rendering import place_label

# Highlighter blue for focus and selection; the user chose it over yellow (2026-09-16).
SELECTION_COLOR = "#00BFFF"


@dataclass(frozen=True)
class LayerStyle:
    """The fixed colours and dash patterns of the prediction layer.

    pred_color is only the fallback for when the caller supplies no class-colour
    lookup. Line widths are not here: the caller passes its own scale-dependent
    width so predictions and annotations stay equally thick at every zoom.
    """
    pred_color: str = "#00BFFF"
    focus_color: str = SELECTION_COLOR
    focus_halo_extra: int = 4
    # Tk on Windows collapses numeric dash lists to one dotted look; these strings stay distinct.
    dash: str = "_"
    rejected_dash: str = "."
    badge_colors: Dict[str, str] = field(default_factory=lambda: {
        "tp": "#4CAF50", "fp": "#EF5350", "fn": "#FFA726"})


def _canvas_points(to_canvas, points):
    flat = []
    for x, y in points:
        cx, cy = to_canvas(x, y)
        flat.extend([cx, cy])
    return flat


def _draw_shape(canvas, to_canvas, kind, points, **kw):
    if kind == "box":
        (x1, y1), (x2, y2) = points
        cx1, cy1 = to_canvas(x1, y1)
        cx2, cy2 = to_canvas(x2, y2)
        return canvas.create_rectangle(cx1, cy1, cx2, cy2, **kw)
    flat = _canvas_points(to_canvas, points)
    if len(flat) < 6:
        return None
    return canvas.create_polygon(*flat, **kw)


def _label_anchor(to_canvas, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return to_canvas(min(xs), min(ys))


def draw_prediction_layer(canvas, to_canvas, state, class_names, font_family,
                          label_size, show_gt, show_pred, style=LayerStyle(),
                          class_color=None, placed_labels=None, line_w=2):
    """Draw predictions, the focused item and the badge. Only the focus gets a label.

    class_color, when given, maps a class id to that class's hex colour.
    placed_labels, when given, is the render pass's shared list of label boxes
    for place_label collision avoidance. line_w is the outline width the caller
    draws annotations with at the current zoom.
    """
    if placed_labels is None:
        placed_labels = []
    focused = None
    if state.queue and 0 <= state.queue_index < len(state.queue):
        focused = state.queue[state.queue_index]
    focused_pred_id = focused.prediction.id if focused and focused.prediction else None
    font = (font_family, label_size, "bold")
    halo_w = line_w + style.focus_halo_extra

    def color_of(class_id):
        return class_color(class_id) if class_color else style.pred_color

    if show_pred:
        for p in state.predictions:
            if p.confidence < state.conf_threshold or p.id == focused_pred_id:
                continue
            verdict = state.verdicts.get(p.id)
            rejected = verdict is not None and verdict.get("action") == "rejected"
            _draw_shape(canvas, to_canvas, p.kind, p.points, outline=color_of(p.class_id),
                        width=line_w, fill="",
                        dash=style.rejected_dash if rejected else style.dash, tags="pred")

    if focused is None:
        return

    ann = focused.annotation if show_gt else None
    pred = focused.prediction if show_pred else None
    selected_id = getattr(state, "_selected_annotation_id", None)
    if pred is not None:
        if ann is None:
            _draw_shape(canvas, to_canvas, pred.kind, pred.points, outline=style.focus_color,
                        width=halo_w, fill="", tags="focus_halo")
        _draw_shape(canvas, to_canvas, pred.kind, pred.points, outline=color_of(pred.class_id),
                    width=line_w + 1, fill="", dash=style.dash, tags="pred_focus")
    if ann is not None and ann.id != selected_id:
        color = color_of(ann.class_id)
        _draw_shape(canvas, to_canvas, ann.kind, ann.points, outline=style.focus_color,
                    width=halo_w, fill="", tags="focus_halo")
        _draw_shape(canvas, to_canvas, ann.kind, ann.points, outline=color,
                    width=line_w, fill="", tags="gt_focus")

    labelled = ann if ann is not None else pred
    if labelled is not None and labelled.id != selected_id:
        name = class_names.get(focused.class_id, str(focused.class_id))
        text = f"{focused.class_id}: {name}"
        if focused.prediction is not None:
            text += f" ({focused.prediction.confidence:.2f})"
        lx, ly = _label_anchor(to_canvas, labelled.points)
        place_label(canvas, placed_labels, lx + 2, ly - 2, text,
                    color_of(focused.class_id), anchor="sw", font=font)

    if show_gt or show_pred:
        verdict = state.verdicts.get(focused.key)
        status = verdict["action"] if verdict else "not reviewed"
        text = f"{focused.kind.upper()}  {status}"
        bfnt = tkFont.Font(family=font_family, size=14, weight="bold")
        tw, th = bfnt.measure(text), bfnt.metrics("linespace")
        cw = canvas.winfo_width() or 800
        bx, by = cw - tw - 20, 10
        canvas.create_rectangle(bx - 6, by - 2, bx + tw + 6, by + th + 4,
                                fill="#1A1A1A", outline="#444444", width=1, tags="badge")
        canvas.create_text(bx, by + 2, anchor="nw", text=text,
                           fill=style.badge_colors.get(focused.kind, "#E0E0E0"),
                           font=(font_family, 14, "bold"), tags="badge")
