"""Focused verdict patterns without creating a Tk window."""

from unittest.mock import Mock

import pytest

from yololabeler.predictions.store import Prediction
from yololabeler.review.engine import QueueItem
from yololabeler.review.layer import LayerStyle, draw_prediction_layer
from yololabeler.state import AppState


@pytest.mark.parametrize("kind,points", [
    ("box", ((10, 10), (50, 50))),
    ("polygon", ((10, 10), (50, 10), (50, 50))),
])
@pytest.mark.parametrize("action", [None, "accepted", "rejected"])
def test_focus_preserves_verdict_pattern_and_adds_halo(monkeypatch, kind, points, action):
    font = Mock()
    font.measure.return_value = 80
    font.metrics.return_value = 16
    monkeypatch.setattr("yololabeler.review.layer.tkFont.Font", Mock(return_value=font))
    monkeypatch.setattr("yololabeler.review.layer.place_label", Mock())
    canvas = Mock()
    canvas.winfo_width.return_value = 640
    state = AppState()
    prediction = Prediction("h:0", kind, points, 0, 0.9, 0)
    state.predictions = [prediction]
    if action:
        state.verdicts[prediction.id] = {"action": action}
    style = LayerStyle()

    def draw():
        canvas.reset_mock()
        draw_prediction_layer(canvas, lambda x, y: (x, y), state, {0: "bur"},
                              "Arial", 10, True, True, style, line_w=4)
        return {call.kwargs["tags"]: call.kwargs
                for method in (canvas.create_rectangle, canvas.create_polygon)
                for call in method.call_args_list}

    unfocused = draw()["pred"]
    state.queue = [QueueItem("fp", prediction, None, None)]
    focused_shapes = draw()
    focused = focused_shapes["pred_focus"]

    assert focused["dash"] == unfocused["dash"]
    assert focused["dash"] == (style.rejected_dash if action == "rejected" else style.dash)
    assert focused["fill"] == ""
    assert focused["width"] == 5
    halo = focused_shapes["focus_halo"]
    assert halo["outline"] == style.focus_color
    assert halo["width"] == 4 + style.focus_halo_extra

    # Revisiting the item after focus moves away keeps the same pattern.
    state.queue = []
    assert draw()["pred"]["dash"] == focused["dash"]
