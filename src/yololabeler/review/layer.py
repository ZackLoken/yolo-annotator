"""Draw the prediction layer and the focused pair on the annotate canvas (spec 4.5).

An annotation is drawn solid, a prediction dashed, a rejected prediction dotted.
While predictions are shown on an image that has them, every shape is coloured
by its review status (state.shape_statuses); otherwise by its class. The review
focus is marked by a highlighter-blue halo under the focused shape rather than
by recolouring it. Needs a Tk canvas; not headless.
"""

from __future__ import annotations

import tkinter.font as tkFont
from dataclasses import dataclass

from yololabeler.rendering import place_label

# Highlighter blue for focus and selection; the user chose it over yellow (2026-09-16).
SELECTION_COLOR = "#00BFFF"
# Green/orange/red for accepted/not reviewed/rejected; the user swapped yellow for orange
# as easier on the eyes (2026-09-17). The hex values are provisional.
STATUS_COLORS = {"accepted": "#00FF00", "not_reviewed": "#FF8C00", "rejected": "#FF0000"}
# The mark a flagged shape's label ends with; drawn alone, white with a black
# halo, on a flagged prediction that has no label of its own. The user chose it (2026-09-16).
FLAG_COLOR = "#FFFFFF"
FLAG_MARK = "?"


def label_text(class_id, class_names, flagged, confidence=None):
    """The label a shape is drawn with: "id: name", the confidence when given, the flag mark when flagged."""
    text = f"{class_id}: {class_names.get(class_id, str(class_id))}"
    if confidence is not None:
        text += f" ({confidence:.2f})"
    if flagged:
        text += f" {FLAG_MARK}"
    return text


def status_colors_active(state, show_pred):
    """The id-to-status map to colour shapes by, or None when they keep class colours."""
    return state.shape_statuses if show_pred else None


def status_color(statuses, shape_id):
    """The status colour for a shape id, treating an id with no entry as not reviewed."""
    return STATUS_COLORS.get(statuses.get(shape_id), STATUS_COLORS["not_reviewed"])


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

    statuses = status_colors_active(state, show_pred)

    def color_of(shape):
        if statuses is not None:
            return status_color(statuses, shape.id)
        return class_color(shape.class_id) if class_color else style.pred_color

    def dash_of(prediction):
        verdict = state.verdicts.get(prediction.id)
        rejected = verdict is not None and verdict.get("action") == "rejected"
        return style.rejected_dash if rejected else style.dash

    if show_pred:
        for p in state.predictions:
            if p.confidence < state.conf_threshold or p.id == focused_pred_id:
                continue
            _draw_shape(canvas, to_canvas, p.kind, p.points, outline=color_of(p),
                        width=line_w, fill="", dash=dash_of(p), tags="pred")
            if p.id in state.flagged_shapes:
                # An unlabelled prediction carries the mark alone where its label would sit.
                lx, ly = _label_anchor(to_canvas, p.points)
                place_label(canvas, placed_labels, lx + 2, ly - 2, FLAG_MARK, FLAG_COLOR,
                            anchor="sw", font=font, tags="flag")

    if focused is None:
        return

    ann = focused.annotation if show_gt else None
    pred = focused.prediction if show_pred else None
    selected_id = getattr(state, "_selected_annotation_id", None)
    if pred is not None:
        if ann is None:
            _draw_shape(canvas, to_canvas, pred.kind, pred.points, outline=style.focus_color,
                        width=halo_w, fill="", tags="focus_halo")
        _draw_shape(canvas, to_canvas, pred.kind, pred.points, outline=color_of(pred),
                    width=line_w + 1, fill="", dash=dash_of(pred), tags="pred_focus")
    if ann is not None and ann.id != selected_id:
        color = color_of(ann)
        _draw_shape(canvas, to_canvas, ann.kind, ann.points, outline=style.focus_color,
                    width=halo_w, fill="", tags="focus_halo")
        _draw_shape(canvas, to_canvas, ann.kind, ann.points, outline=color,
                    width=line_w, fill="", tags="gt_focus")

    labelled = ann if ann is not None else pred
    if labelled is not None and labelled.id != selected_id:
        confidence = focused.prediction.confidence if focused.prediction is not None else None
        text = label_text(focused.class_id, class_names, labelled.id in state.flagged_shapes,
                          confidence)
        lx, ly = _label_anchor(to_canvas, labelled.points)
        place_label(canvas, placed_labels, lx + 2, ly - 2, text,
                    color_of(labelled), anchor="sw", font=font)

    if show_gt or show_pred:
        verdict = state.verdicts.get(focused.key)
        status = verdict["action"] if verdict else "not_reviewed"
        text = f"{focused.kind.upper()}  {status.replace('_', ' ')}"
        if focused.key in state.flag_markers or (
                focused.annotation is not None and focused.annotation.id in state.flag_markers):
            text += "  flagged"
        bfnt = tkFont.Font(family=font_family, size=14, weight="bold")
        tw, th = bfnt.measure(text), bfnt.metrics("linespace")
        cw = canvas.winfo_width() or 800
        bx, by = cw - tw - 20, 10
        canvas.create_rectangle(bx - 6, by - 2, bx + tw + 6, by + th + 4,
                                fill="#1A1A1A", outline="#444444", width=1, tags="badge")
        canvas.create_text(bx, by + 2, anchor="nw", text=text,
                           fill=STATUS_COLORS.get(status, STATUS_COLORS["not_reviewed"]),
                           font=(font_family, 14, "bold"), tags="badge")
