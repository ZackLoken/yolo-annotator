"""Tests for yololabeler.rendering, halo text drawn on a real Tk canvas."""

import tkinter as tk

import pytest

from yololabeler.rendering import (
    halo_text, place_label, rounded_rect, _cached_font, _font_cache, _HALO_OFFSETS,
    _LABEL_NUDGE,
)


@pytest.fixture(scope="module")
def tk_root():
    """One hidden Tk root for the module; skipped when there is no display."""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvas(tk_root):
    """A real Tk Canvas, empty at the start of each test."""
    c = tk.Canvas(tk_root, width=200, height=200)
    yield c
    c.destroy()


# ── _HALO_OFFSETS ───────────────────────────────────────────────────────────

class TestHaloOffsets:
    def test_offsets_are_unique(self):
        assert len(set(_HALO_OFFSETS)) == len(_HALO_OFFSETS)

    def test_centre_is_not_an_offset(self):
        assert (0, 0) not in _HALO_OFFSETS

    def test_offsets_stay_within_two_pixels(self):
        assert all(abs(dx) <= 2 and abs(dy) <= 2 for dx, dy in _HALO_OFFSETS)


# ── halo_text ───────────────────────────────────────────────────────────────

class TestHaloText:
    def test_item_count(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        assert len(canvas.find_all()) == len(_HALO_OFFSETS) + 1

    def test_all_items_share_the_text(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        assert all(canvas.itemcget(i, "text") == "hello"
                   for i in canvas.find_all())

    def test_halo_items_are_black(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        items = canvas.find_all()
        assert all(canvas.itemcget(i, "fill") == "black" for i in items[:-1])

    def test_last_item_uses_the_fill_colour(self, canvas):
        halo_text(canvas, 50, 60, "hello", "#ff8800")
        last = canvas.find_all()[-1]
        assert canvas.itemcget(last, "fill") == "#ff8800"

    def test_last_item_sits_on_the_requested_point(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        last = canvas.find_all()[-1]
        assert canvas.coords(last) == [50.0, 60.0]

    def test_halo_positions_match_the_offsets(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        items = canvas.find_all()
        positions = {tuple(canvas.coords(i)) for i in items[:-1]}
        expected = {(50.0 + dx, 60.0 + dy) for dx, dy in _HALO_OFFSETS}
        assert positions == expected

    def test_keyword_arguments_are_forwarded(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white", anchor="nw")
        assert all(canvas.itemcget(i, "anchor") == "nw"
                   for i in canvas.find_all())

    def test_float_coordinates_accepted(self, canvas):
        halo_text(canvas, 10.5, 20.5, "x", "white")
        last = canvas.find_all()[-1]
        assert canvas.coords(last) == pytest.approx([10.5, 20.5])

    def test_repeated_calls_accumulate(self, canvas):
        halo_text(canvas, 10, 10, "a", "white")
        halo_text(canvas, 20, 20, "b", "white")
        assert len(canvas.find_all()) == 2 * (len(_HALO_OFFSETS) + 1)

    def test_empty_text_still_draws_items(self, canvas):
        halo_text(canvas, 10, 10, "", "white")
        assert len(canvas.find_all()) == len(_HALO_OFFSETS) + 1


# ── rounded_rect ──────────────────────────────────────────────────────────

class TestRoundedRect:
    def test_is_one_smoothed_polygon_spanning_the_box(self, canvas):
        item = rounded_rect(canvas, 10, 20, 110, 70, fill="black", outline="white", tags="p")
        assert canvas.type(item) == "polygon"
        assert canvas.itemcget(item, "smooth") in ("1", "true")
        assert canvas.itemcget(item, "fill") == "black"
        assert canvas.itemcget(item, "outline") == "white"
        assert "p" in canvas.gettags(item)
        x0, y0, x1, y1 = canvas.bbox(item)
        assert x0 <= 10 and y0 <= 20 and x1 >= 110 and y1 >= 70

    def test_sides_stay_straight_between_the_corners(self, canvas):
        item = rounded_rect(canvas, 0, 0, 100, 50, radius=6)
        coords = canvas.coords(item)
        xs, ys = coords[0::2], coords[1::2]
        assert min(xs) == 0 and max(xs) == 100 and min(ys) == 0 and max(ys) == 50
        assert xs.count(0) >= 3 and xs.count(100) >= 3

    def test_radius_is_clamped_to_half_the_shorter_side(self, canvas):
        item = rounded_rect(canvas, 0, 0, 100, 8, radius=20)
        ys = canvas.coords(item)[1::2]
        assert min(ys) == 0 and max(ys) == 8


# ── place_label ───────────────────────────────────────────────────────────

class TestPlaceLabel:
    FONT = ("Arial", 10)

    def test_first_label_is_not_nudged(self, canvas):
        placed = []
        place_label(canvas, placed, 50, 60, "hello", "white", font=self.FONT)
        assert len(placed) == 1
        last = canvas.find_all()[-1]
        assert canvas.coords(last) == [50.0, 60.0]

    def test_second_label_at_same_anchor_is_nudged_and_agrees_with_its_box(self, canvas):
        placed = []
        place_label(canvas, placed, 50, 60, "hello", "white", font=self.FONT)
        place_label(canvas, placed, 50, 60, "hello", "white", font=self.FONT)
        assert len(placed) == 2
        last = canvas.find_all()[-1]
        drawn_y = canvas.coords(last)[1]
        assert drawn_y >= 60 + _LABEL_NUDGE
        # anchor "sw": box_at sets y1 to the y actually drawn, so the two must agree
        assert placed[1][3] == pytest.approx(drawn_y)

    def test_two_labels_far_apart_are_both_unnudged(self, canvas):
        placed = []
        place_label(canvas, placed, 10, 10, "a", "white", font=self.FONT)
        place_label(canvas, placed, 500, 500, "b", "white", font=self.FONT)
        n = len(_HALO_OFFSETS) + 1
        items = canvas.find_all()
        assert canvas.coords(items[n - 1]) == [10.0, 10.0]
        assert canvas.coords(items[-1]) == [500.0, 500.0]
        assert len(placed) == 2


# ── _cached_font ──────────────────────────────────────────────────────────

class TestCachedFontStaleInterpreter:
    """A cached Font is bound to the Tk interpreter live when it was built; a
    stale one (its interpreter torn down) must be rebuilt, not raise."""

    FONT = ("Arial", 11, "bold")

    def test_rebuilds_when_the_cached_font_reports_a_destroyed_interpreter(self, tk_root, monkeypatch):
        _font_cache.pop(self.FONT, None)
        first = _cached_font(self.FONT)
        first.metrics("linespace")  # sanity check: a real, working Font

        def stale_metrics(*args, **kwargs):
            raise tk.TclError('can\'t invoke "font" command: application has been destroyed')
        monkeypatch.setattr(first, "metrics", stale_metrics)

        second = _cached_font(self.FONT)  # must not raise despite the stale cache hit
        assert second is not first
        assert _font_cache[self.FONT] is second
        assert second.metrics("linespace") > 0  # the rebuilt Font is real and usable

        _font_cache.pop(self.FONT, None)
