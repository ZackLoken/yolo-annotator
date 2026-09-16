"""Tests for yololabeler.review.layer on a real Tk canvas; skipped without a display."""

import tkinter as tk

import pytest

from yololabeler.annotation.document import new_annotation
from yololabeler.predictions.store import Prediction
from yololabeler.review.engine import QueueItem
from yololabeler.review.layer import (
    SELECTION_COLOR, STATUS_COLORS, LayerStyle, draw_prediction_layer,
)
from yololabeler.state import AppState


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvas(tk_root):
    c = tk.Canvas(tk_root, width=300, height=300)
    yield c
    c.destroy()


def ident(ix, iy):
    return ix, iy


def red(class_id):
    return "#FF0000"


def make_state():
    s = AppState()
    s.conf_threshold = 0.5
    s.predictions = [Prediction("h:0", "box", ((10, 10), (50, 50)), 0, 0.9, 0),
                     Prediction("h:1", "box", ((100, 100), (150, 150)), 0, 0.3, 1),
                     Prediction("h:2", "polygon", ((200, 200), (250, 200), (250, 250)), 0, 0.8, 2)]
    return s


def texts(canvas):
    return [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]


def label_texts(canvas):
    """Distinct label strings, ignoring the badge and the halo copies of each label."""
    badge = {canvas.itemcget(i, "text") for i in canvas.find_withtag("badge")
             if canvas.type(i) == "text"}
    return set(texts(canvas)) - badge


# ── draw_prediction_layer ───────────────────────────────────────────────────

class TestDrawPredictionLayer:
    def test_low_confidence_not_drawn(self, canvas):
        s = make_state()
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        kinds = [canvas.type(i) for i in canvas.find_all()]
        assert kinds.count("rectangle") == 1 and kinds.count("polygon") == 1

    def test_show_pred_off_draws_nothing(self, canvas):
        s = make_state()
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, False)
        assert canvas.find_all() == ()

    def test_predictions_are_dashed_at_the_callers_line_width(self, canvas):
        s = make_state()
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True, line_w=5)
        for item in canvas.find_withtag("pred"):
            assert float(canvas.itemcget(item, "width")) == 5.0
            assert canvas.itemcget(item, "dash") != ""

    def test_prediction_uses_the_class_colour_untinted(self, canvas):
        s = make_state()
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        rect = [i for i in canvas.find_withtag("pred") if canvas.type(i) == "rectangle"][0]
        assert canvas.itemcget(rect, "outline") == "#FF0000"

    def test_without_a_class_colour_predictions_keep_the_fallback(self, canvas):
        s = make_state()
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        rect = [i for i in canvas.find_all() if canvas.type(i) == "rectangle"][0]
        assert canvas.itemcget(rect, "outline") == LayerStyle().pred_color

    def test_accepted_prediction_stays_drawn(self, canvas):
        s = make_state()
        s.verdicts = {"h:0": {"action": "accepted"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        kinds = [canvas.type(i) for i in canvas.find_withtag("pred")]
        assert kinds.count("rectangle") == 1 and kinds.count("polygon") == 1

    def test_rejected_prediction_uses_the_rejected_dash_and_no_fill(self, canvas):
        s = make_state()
        s.verdicts = {"h:0": {"action": "rejected"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        rect = [i for i in canvas.find_withtag("pred") if canvas.type(i) == "rectangle"][0]
        assert canvas.itemcget(rect, "dash") == LayerStyle().rejected_dash
        assert canvas.itemcget(rect, "fill") == ""
        assert canvas.itemcget(rect, "stipple") == ""

    def test_focused_fp_gets_a_blue_halo_in_class_colour_with_one_label(self, canvas):
        s = make_state()
        s.queue = [QueueItem("fp", s.predictions[0], None, None)]
        s.queue_index = 0
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        halo = canvas.find_withtag("focus_halo")
        assert len(halo) == 1 and canvas.itemcget(halo[0], "outline") == SELECTION_COLOR
        focus = canvas.find_withtag("pred_focus")
        assert len(focus) == 1 and canvas.itemcget(focus[0], "outline") == "#FF0000"
        assert label_texts(canvas) == {"0: burr (0.90)"}
        assert "FP  not reviewed" in texts(canvas)

    def test_focused_rejected_prediction_keeps_the_rejected_dash(self, canvas):
        s = make_state()
        s.queue = [QueueItem("fp", s.predictions[0], None, None)]
        s.queue_index = 0
        s.verdicts = {"h:0": {"action": "rejected"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        focus = canvas.find_withtag("pred_focus")
        assert len(focus) == 1
        assert canvas.itemcget(focus[0], "dash") == LayerStyle().rejected_dash

    def test_focused_pair_halos_the_gt_and_shows_one_label(self, canvas):
        s = make_state()
        ann = new_annotation("box", ((12, 12), (52, 52)), 0, "z")
        s.queue = [QueueItem("tp", s.predictions[0], ann, 0.9)]
        s.queue_index = 0
        s.verdicts = {"h:0": {"action": "accepted"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        gt = canvas.find_withtag("gt_focus")
        assert len(gt) == 1 and canvas.itemcget(gt[0], "outline") == "#FF0000"
        halo = canvas.find_withtag("focus_halo")
        assert len(halo) == 1 and canvas.coords(halo[0]) == canvas.coords(gt[0])
        assert label_texts(canvas) == {"0: burr (0.90)"}
        assert "TP  accepted" in texts(canvas)

    def test_focused_gt_is_left_to_the_selection_drawing_when_selected(self, canvas):
        s = make_state()
        ann = new_annotation("box", ((12, 12), (52, 52)), 0, "z")
        s.queue = [QueueItem("tp", s.predictions[0], ann, 0.9)]
        s.queue_index = 0
        s._selected_annotation_id = ann.id
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        assert canvas.find_withtag("gt_focus") == ()
        assert label_texts(canvas) == set()

    def test_focused_prediction_not_drawn_when_show_pred_off(self, canvas):
        s = make_state()
        s.queue = [QueueItem("fp", s.predictions[0], None, None)]
        s.queue_index = 0
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, False, False)
        assert canvas.find_withtag("pred_focus") == ()
        assert canvas.find_all() == ()


# ── status colours ──────────────────────────────────────────────────────────

def badge_fill(canvas):
    return [canvas.itemcget(i, "fill") for i in canvas.find_withtag("badge")
            if canvas.type(i) == "text"][0]


class TestStatusColours:
    @pytest.mark.parametrize("verdicts, status", [
        ({}, "not_reviewed"),
        ({"h:0": {"action": "accepted"}}, "accepted"),
        ({"h:0": {"action": "rejected"}}, "rejected")])
    def test_badge_text_takes_the_status_colour(self, canvas, verdicts, status):
        s = make_state()
        s.queue = [QueueItem("fp", s.predictions[0], None, None)]
        s.queue_index = 0
        s.verdicts = verdicts
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        assert badge_fill(canvas) == STATUS_COLORS[status]

    def test_predictions_take_the_status_colour_over_the_class_colour(self, canvas):
        s = make_state()
        s.shape_statuses = {"h:0": "accepted", "h:2": "rejected"}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        outlines = {canvas.type(i): canvas.itemcget(i, "outline")
                    for i in canvas.find_withtag("pred")}
        assert outlines == {"rectangle": STATUS_COLORS["accepted"],
                            "polygon": STATUS_COLORS["rejected"]}

    def test_an_id_with_no_status_is_drawn_as_not_reviewed(self, canvas):
        s = make_state()
        s.shape_statuses = {}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        for item in canvas.find_withtag("pred"):
            assert canvas.itemcget(item, "outline") == STATUS_COLORS["not_reviewed"]

    def test_focused_pair_takes_the_status_colour(self, canvas):
        s = make_state()
        ann = new_annotation("box", ((12, 12), (52, 52)), 0, "z")
        s.queue = [QueueItem("tp", s.predictions[0], ann, 0.9)]
        s.queue_index = 0
        s.shape_statuses = {"h:0": "not_reviewed", ann.id: "not_reviewed"}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True,
                              LayerStyle(), red)
        gt = canvas.find_withtag("gt_focus")
        assert canvas.itemcget(gt[0], "outline") == STATUS_COLORS["not_reviewed"]
        pred = canvas.find_withtag("pred_focus")
        assert canvas.itemcget(pred[0], "outline") == STATUS_COLORS["not_reviewed"]
