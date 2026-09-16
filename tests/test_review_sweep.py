"""Review navigation checks without creating a Tk window."""

from unittest.mock import Mock

import pytest
from PIL import Image

from yololabeler.annotation.document import Document
from yololabeler.annotation.tab import AnnotateTab
from yololabeler.gui import YoloLabeler
from yololabeler.predictions.store import Prediction
from yololabeler.review.engine import ReviewEngine
from yololabeler.review.panel import ReviewPanel
from yololabeler.state import AppState


@pytest.fixture
def app():
    app = YoloLabeler.__new__(YoloLabeler)
    app._state = AppState()
    app.images = ["one.jpg", "two.jpg"]
    app.document = Document("one.jpg", 640, 480)
    app.predictions = [Prediction("h:0", "box", ((10, 10), (50, 50)), 0, 0.9, 0)]
    app._review = ReviewEngine(app._state)
    app.verdicts = app._review.verdicts("one.jpg")
    app._review_panel = ReviewPanel(app)
    app._review_panel.update_labels = Mock()
    app._annotate_tab = Mock()
    app._engine = Mock()
    app.save_current = Mock(return_value=None)
    app.go_to_image = Mock()
    app._select_class_by_id = Mock()
    app._set_mode = Mock()
    app._review_panel.refresh()
    return app


@pytest.mark.parametrize("action", ["accept_item", "reject_item"])
@pytest.mark.parametrize("status_filter", ["all", "not_reviewed", "reviewed"])
def test_last_verdict_fits_image_without_advancing(app, action, status_filter):
    if status_filter == "reviewed":
        app.verdicts["h:0"] = {"action": "rejected"}
    app._review_status_filter = status_filter
    app._review_panel.refresh()

    getattr(app, action)()

    app._annotate_tab.fit_to_window.assert_called_once_with()
    app._annotate_tab.zoom_to_bbox.assert_not_called()
    app.go_to_image.assert_not_called()
    assert app.index == 0
    assert "one.jpg" not in app._completed_images
    assert app.verdicts["h:0"]["action"] == (
        "accepted" if action == "accept_item" else "rejected")


def test_remaining_prediction_still_gets_focus(app):
    app.predictions.append(Prediction("h:1", "box", ((80, 80), (100, 100)), 0, 0.9, 1))
    app._review_panel.refresh()

    app.reject_item()

    assert app._review_panel.current_item().key == "h:1"
    app._annotate_tab.zoom_to_bbox.assert_called_once_with(80, 80, 100, 100)
    app._annotate_tab.fit_to_window.assert_not_called()
    app.go_to_image.assert_not_called()


def test_image_without_predictions_opens_fitted_and_stays(app, tmp_path):
    Image.new("RGB", (640, 480)).save(tmp_path / "one.jpg")
    app.image_folder = str(tmp_path)
    app.state_dir = str(tmp_path)
    app.predictions = []
    app._review_panel.load_predictions_for_current_image = Mock()
    app.update_title = Mock()
    app._update_status = Mock()
    tab = app._annotate_tab
    tab.app = app

    def load_document():
        app.document = Document("one.jpg", 640, 480)
        return []

    tab.load_document_for_current_image.side_effect = load_document
    AnnotateTab.load_image(tab)

    tab.fit_to_window.assert_called_once_with()
    tab.zoom_to_bbox.assert_not_called()
    app.go_to_image.assert_not_called()
    assert app.queue == []
    assert app.index == 0
