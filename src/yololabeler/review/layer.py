"""Draw the prediction layer and the focused pair on the annotate canvas (spec 4.5)."""

from __future__ import annotations

import tkinter.font as tkFont
from dataclasses import dataclass, field
from typing import Dict

from yololabeler.rendering import halo_text

_ANNOTATED_ACTIONS = ("accepted", "confirmed")


@dataclass(frozen=True)
class LayerStyle:
    """Colours and widths from spec 8; do not add colours here."""
    pred_color: str = "#00BFFF"
    focused_gt_color: str = "#FFD700"
    reviewed_stipple: str = "gray12"
    line_w: int = 2
    focused_w: int = 3
    dash: tuple = (4, 3)
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
                          label_size, show_gt, show_pred, style=LayerStyle()):
    """Draw predictions, the focused pair and the badge. Only the focus gets labels."""
    focused = None
    if state.queue and 0 <= state.queue_index < len(state.queue):
        focused = state.queue[state.queue_index]
    focused_pred_id = focused.prediction.id if focused and focused.prediction else None
    font = (font_family, label_size, "bold")

    if show_pred:
        for p in state.predictions:
            if p.confidence < state.conf_threshold or p.id == focused_pred_id:
                continue
            verdict = state.verdicts.get(p.id)
            if verdict and verdict.get("action") in _ANNOTATED_ACTIONS:
                continue
            reviewed = verdict is not None
            _draw_shape(canvas, to_canvas, p.kind, p.points,
                        outline=style.pred_color, width=style.line_w, dash=style.dash,
                        fill=style.pred_color if reviewed else "",
                        stipple=style.reviewed_stipple if reviewed else "",
                        tags="pred")
        if focused and focused.prediction is not None:
            p = focused.prediction
            _draw_shape(canvas, to_canvas, p.kind, p.points,
                        outline=style.pred_color, width=style.focused_w, fill="",
                        tags="pred_focus")
            lx, ly = _label_anchor(to_canvas, p.points)
            name = class_names.get(p.class_id, str(p.class_id))
            halo_text(canvas, lx + 2, ly - 2, f"Pred {p.class_id}: {name} ({p.confidence:.2f})",
                      style.pred_color, anchor="sw", font=font)

    if show_gt and focused and focused.annotation is not None:
        a = focused.annotation
        _draw_shape(canvas, to_canvas, a.kind, a.points,
                    outline=style.focused_gt_color, width=style.focused_w, fill="",
                    tags="gt_focus")
        lx, ly = _label_anchor(to_canvas, a.points)
        name = class_names.get(a.class_id, str(a.class_id))
        halo_text(canvas, lx + 2, ly + 2 + label_size * 2, f"GT {a.class_id}: {name}",
                  style.focused_gt_color, anchor="nw", font=font)

    if focused and (show_gt or show_pred):
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
