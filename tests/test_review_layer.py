"""Tests for yololabeler.review.layer on a real Tk canvas; skipped without a display."""

import tkinter as tk

import pytest

from yololabeler.annotation.document import new_annotation
from yololabeler.predictions.store import Prediction
from yololabeler.review.engine import QueueItem
from yololabeler.review.layer import LayerStyle, draw_prediction_layer
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


def make_state():
    s = AppState()
    s.conf_threshold = 0.5
    s.predictions = [Prediction("h:0", "box", ((10, 10), (50, 50)), 0, 0.9, 0),
                     Prediction("h:1", "box", ((100, 100), (150, 150)), 0, 0.3, 1),
                     Prediction("h:2", "polygon", ((200, 200), (250, 200), (250, 250)), 0, 0.8, 2)]
    return s


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

    def test_focused_fp_gets_thick_outline_label_and_badge(self, canvas):
        s = make_state()
        s.queue = [QueueItem("fp", s.predictions[0], None, None)]
        s.queue_index = 0
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        widths = {float(canvas.itemcget(i, "width")) for i in canvas.find_withtag("pred_focus")}
        assert widths == {3.0}
        texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
        assert "Pred 0: burr (0.90)" in texts
        assert "FP  not reviewed" in texts

    def test_focused_pair_draws_gt_in_focus_colour(self, canvas):
        s = make_state()
        ann = new_annotation("box", ((12, 12), (52, 52)), 0, "z")
        s.queue = [QueueItem("tp", s.predictions[0], ann, 0.9)]
        s.queue_index = 0
        s.verdicts = {"h:0": {"action": "confirmed"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        gt = canvas.find_withtag("gt_focus")
        assert len(gt) == 1 and canvas.itemcget(gt[0], "outline") == LayerStyle().focused_gt_color
        texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
        assert "TP  confirmed" in texts

    def test_reviewed_prediction_is_stippled(self, canvas):
        s = make_state()
        s.verdicts = {"h:0": {"action": "rejected"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        rect = [i for i in canvas.find_all() if canvas.type(i) == "rectangle"][0]
        assert canvas.itemcget(rect, "stipple") == "gray12"

    def test_accepted_prediction_is_not_drawn(self, canvas):
        s = make_state()
        s.verdicts = {"h:0": {"action": "accepted"}}
        draw_prediction_layer(canvas, ident, s, {0: "burr"}, "Arial", 9, True, True)
        kinds = [canvas.type(i) for i in canvas.find_all()]
