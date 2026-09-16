"""Prediction drawing colors without creating a Tk window."""

from unittest.mock import Mock

import pytest

from yololabeler.annotation.document import new_annotation
from yololabeler.predictions.store import Prediction
from yololabeler.review.engine import QueueItem
from yololabeler.review.layer import (
    ACCEPTED_COLOR, REJECTED_COLOR, SELECTION_COLOR, UNREVIEWED_COLOR,
    LayerStyle, draw_prediction_layer,
)
from yololabeler.state import AppState


@pytest.fixture
def canvas(monkeypatch):
    font = Mock()
    font.measure.return_value = 80
    font.metrics.return_value = 16
    monkeypatch.setattr("yololabeler.review.layer.tkFont.Font", Mock(return_value=font))
    monkeypatch.setattr("yololabeler.review.layer.place_label", Mock())
    canvas = Mock()
    canvas.winfo_width.return_value = 640
    return canvas


def draw(canvas, state):
    draw_prediction_layer(canvas, lambda x, y: (x, y), state, {0: "bur"},
                          "Arial", 10, True, True, class_color=lambda cid: "#FF00FF")


def shapes(canvas, tag):
    return [call.kwargs for method in (canvas.create_rectangle, canvas.create_polygon)
            for call in method.call_args_list if call.kwargs.get("tags") == tag]


@pytest.mark.parametrize("kind,points", [
    ("box", ((10, 10), (50, 50))),
    ("polygon", ((10, 10), (50, 10), (50, 50))),
])
@pytest.mark.parametrize("action,color", [
    (None, UNREVIEWED_COLOR), ("accepted", ACCEPTED_COLOR), ("rejected", REJECTED_COLOR),
])
@pytest.mark.parametrize("focused", [False, True])
def test_prediction_color_follows_verdict(canvas, kind, points, action, color, focused):
    state = AppState()
    prediction = Prediction("h:0", kind, points, 0, 0.9, 0)
    state.predictions = [prediction]
    if action:
        state.verdicts[prediction.id] = {"action": action}
    if focused:
        state.queue = [QueueItem("fp", prediction, None, None)]

    draw(canvas, state)

    prediction_shapes = shapes(canvas, "pred_focus" if focused else "pred")
    assert len(prediction_shapes) == 1
    assert prediction_shapes[0]["outline"] == color
    assert prediction_shapes[0]["fill"] == ""
    if focused:
        assert shapes(canvas, "focus_halo")[0]["outline"] == SELECTION_COLOR
    else:
        assert prediction_shapes[0]["dash"] == (
            LayerStyle().rejected_dash if action == "rejected" else LayerStyle().dash)


def test_matched_annotation_keeps_class_color(canvas):
    state = AppState()
    points = ((10, 10), (50, 50))
    prediction = Prediction("h:0", "box", points, 0, 0.9, 0)
    annotation = new_annotation("box", points, 0, "reviewer")
    state.predictions = [prediction]
    state.queue = [QueueItem("tp", prediction, annotation, 1.0)]
    state.verdicts[prediction.id] = {"action": "accepted"}

    draw(canvas, state)

    assert shapes(canvas, "gt_focus")[0]["outline"] == "#FF00FF"
    assert shapes(canvas, "pred_focus")[0]["outline"] == ACCEPTED_COLOR
    assert shapes(canvas, "focus_halo")[0]["outline"] == SELECTION_COLOR


def test_low_confidence_prediction_stays_hidden(canvas):
    state = AppState()
    state.predictions = [Prediction("h:0", "box", ((10, 10), (50, 50)), 0, 0.1, 0)]
    state.verdicts["h:0"] = {"action": "accepted"}
    draw(canvas, state)
    assert shapes(canvas, "pred") == []
