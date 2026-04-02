"""Tests for yololabeler.matching — geometry helpers and matching engine."""

import pytest

from yololabeler.matching import (
    point_to_segment_dist, point_in_polygon,
    box_iou, polygon_iou, box_to_points, compute_matches,
)
from shapely.geometry import Polygon as ShapelyPolygon


# ── point_to_segment_dist ───────────────────────────────────────────────────

class TestPointToSegmentDist:
    def test_perpendicular(self):
        # Point directly above midpoint of horizontal segment
        assert pytest.approx(point_to_segment_dist(5, 5, 0, 0, 10, 0)) == 5.0

    def test_at_endpoint(self):
        # Point closest to endpoint A
        d = point_to_segment_dist(0, 5, 0, 0, 10, 0)
        assert pytest.approx(d) == 5.0

    def test_beyond_endpoint(self):
        # Point past endpoint B
        d = point_to_segment_dist(15, 0, 0, 0, 10, 0)
        assert pytest.approx(d) == 5.0

    def test_zero_length_segment(self):
        d = point_to_segment_dist(3, 4, 0, 0, 0, 0)
        assert pytest.approx(d) == 5.0

    def test_on_segment(self):
        d = point_to_segment_dist(5, 0, 0, 0, 10, 0)
        assert pytest.approx(d) == 0.0


# ── point_in_polygon ────────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_inside_square(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon(5, 5, sq) is True

    def test_outside_square(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon(15, 5, sq) is False

    def test_inside_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon(5, 3, tri) is True

    def test_outside_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon(0, 10, tri) is False


# ── box_iou ─────────────────────────────────────────────────────────────────

class TestBoxIou:
    def test_perfect_overlap(self):
        b = (0, 0, 10, 10, 0)
        assert pytest.approx(box_iou(b, b)) == 1.0

    def test_no_overlap(self):
        b1 = (0, 0, 10, 10, 0)
        b2 = (20, 20, 30, 30, 0)
        assert box_iou(b1, b2) == 0.0

    def test_partial_overlap(self):
        b1 = (0, 0, 10, 10, 0)
        b2 = (5, 5, 15, 15, 0)
        # intersection 5×5=25, union 100+100-25=175
        assert pytest.approx(box_iou(b1, b2)) == 25.0 / 175.0

    def test_zero_area(self):
        b1 = (5, 5, 5, 5, 0)
        b2 = (0, 0, 10, 10, 0)
        assert box_iou(b1, b2) == 0.0


# ── polygon_iou ─────────────────────────────────────────────────────────────

class TestPolygonIou:
    def test_identical(self):
        g = ShapelyPolygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert pytest.approx(polygon_iou(g, g.area, g, g.area)) == 1.0

    def test_no_overlap(self):
        g1 = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        g2 = ShapelyPolygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        assert polygon_iou(g1, g1.area, g2, g2.area) == 0.0


# ── box_to_points ───────────────────────────────────────────────────────────

class TestBoxToPoints:
    def test_basic(self):
        pts = box_to_points((10, 20, 30, 40, 0))
        assert pts == [(10, 20), (30, 20), (30, 40), (10, 40)]


# ── compute_matches ─────────────────────────────────────────────────────────

class TestComputeMatches:
    def test_perfect_match_boxes(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        result = compute_matches(gt, [], pred, [], iou_threshold=0.5)
        assert len(result['tp']) == 1
        assert len(result['fp']) == 0
        assert len(result['fn']) == 0

    def test_no_match_different_class(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 1, 0.9)]  # class 1 ≠ class 0
        result = compute_matches(gt, [], pred, [], iou_threshold=0.5)
        assert len(result['tp']) == 0
        assert len(result['fp']) == 1
        assert len(result['fn']) == 1

    def test_below_conf_threshold(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.1)]
        result = compute_matches(gt, [], pred, [], conf_threshold=0.25)
        assert len(result['tp']) == 0
        assert len(result['fp']) == 0  # filtered out
        assert len(result['fn']) == 1

    def test_below_iou_threshold(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(8, 8, 18, 18, 0, 0.9)]  # IoU ≈ 0.02
        result = compute_matches(gt, [], pred, [], iou_threshold=0.5)
        assert len(result['tp']) == 0
        assert len(result['fp']) == 1
        assert len(result['fn']) == 1

    def test_empty(self):
        result = compute_matches([], [], [], [])
        assert result == {'tp': [], 'fp': [], 'fn': []}

    def test_polygon_match(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        gt_poly = [(pts, 0)]
        pred_poly = [(pts, 0, 0.9)]
        result = compute_matches([], gt_poly, [], pred_poly, iou_threshold=0.5)
        assert len(result['tp']) == 1
        assert len(result['fp']) == 0
        assert len(result['fn']) == 0

    def test_greedy_best_iou_wins(self):
        """When two predictions match the same GT, the higher-IoU pair wins."""
        gt = [(0, 0, 10, 10, 0)]
        pred = [
            (0, 0, 10, 10, 0, 0.9),   # perfect IoU=1.0
            (1, 1, 11, 11, 0, 0.95),   # good IoU but lower
        ]
        result = compute_matches(gt, [], pred, [], iou_threshold=0.3)
        assert len(result['tp']) == 1
        assert len(result['fp']) == 1
        # TP should be the perfect match (pred index 0)
        assert result['tp'][0][3] == 0  # pred_idx
