"""Tests for yololabeler.matching — geometry helpers and matching engine."""

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from yololabeler.matching import (
    box_iou,
    box_to_points,
    compute_matches,
    point_in_polygon,
    point_to_segment_dist,
    polygon_area,
    polygon_iou,
)

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
        # intersection 5x5=25, union 100+100-25=175
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


# ── polygon_area ────────────────────────────────────────────────────────────


class TestPolygonArea:
    def test_square(self):
        assert polygon_area(
            [(0, 0), (10, 0), (10, 10), (0, 10)]
        ) == pytest.approx(100)

    def test_triangle_in_either_winding(self):
        assert polygon_area([(0, 0), (10, 0), (0, 10)]) == pytest.approx(50)
        assert polygon_area([(0, 0), (0, 10), (10, 0)]) == pytest.approx(50)

    def test_collinear_and_short_are_zero(self):
        assert polygon_area([(0, 0), (5, 5), (10, 10)]) == 0
        assert polygon_area([(0, 0), (5, 5)]) == 0


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
        assert len(result["tp"]) == 1
        assert len(result["fp"]) == 0
        assert len(result["fn"]) == 0

    def test_no_match_different_class(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 1, 0.9)]  # class 1 ≠ class 0
        result = compute_matches(gt, [], pred, [], iou_threshold=0.5)
        assert len(result["tp"]) == 0
        assert len(result["fp"]) == 1
        assert len(result["fn"]) == 1

    def test_below_conf_threshold(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.1)]
        result = compute_matches(gt, [], pred, [], conf_threshold=0.25)
        assert len(result["tp"]) == 0
        assert len(result["fp"]) == 0  # filtered out
        assert len(result["fn"]) == 1

    def test_below_iou_threshold(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(8, 8, 18, 18, 0, 0.9)]  # IoU ≈ 0.02
        result = compute_matches(gt, [], pred, [], iou_threshold=0.5)
        assert len(result["tp"]) == 0
        assert len(result["fp"]) == 1
        assert len(result["fn"]) == 1

    def test_empty(self):
        result = compute_matches([], [], [], [])
        assert result == {"tp": [], "fp": [], "fn": []}

    def test_polygon_match(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        gt_poly = [(pts, 0)]
        pred_poly = [(pts, 0, 0.9)]
        result = compute_matches([], gt_poly, [], pred_poly, iou_threshold=0.5)
        assert len(result["tp"]) == 1
        assert len(result["fp"]) == 0
        assert len(result["fn"]) == 0

    def test_a_box_never_matches_a_polygon(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = compute_matches(
            [], [(pts, 0)], [(0, 0, 10, 10, 0, 0.9)], [], iou_threshold=0.5
        )
        assert (
            result["tp"] == []
            and len(result["fp"]) == 1
            and len(result["fn"]) == 1
        )
        result = compute_matches(
            [(0, 0, 10, 10, 0)], [], [], [(pts, 0, 0.9)], iou_threshold=0.5
        )
        assert (
            result["tp"] == []
            and len(result["fp"]) == 1
            and len(result["fn"]) == 1
        )

    def test_greedy_best_iou_wins(self):
        """When two predictions match the same GT, the higher-IoU pair wins."""
        gt = [(0, 0, 10, 10, 0)]
        pred = [
            (0, 0, 10, 10, 0, 0.9),  # perfect IoU=1.0
            (1, 1, 11, 11, 0, 0.95),  # good IoU but lower
        ]
        result = compute_matches(gt, [], pred, [], iou_threshold=0.3)
        assert len(result["tp"]) == 1
        assert len(result["fp"]) == 1
        # TP should be the perfect match (pred index 0)
        assert result["tp"][0][3] == 0  # pred_idx


# --- compute_matches links ---


class TestComputeMatchesLinks:
    def test_link_pairs_across_classes(self):
        gt = [(0, 0, 10, 10, 1)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        links = [(("box", 0), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert len(result["tp"]) == 1
        assert result["fp"] == [] and result["fn"] == []
        assert result["tp"][0][5] == 1  # the annotation's class

    def test_link_pairs_below_the_iou_threshold(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(8, 8, 18, 18, 0, 0.9)]
        links = [(("box", 0), ("box", 0))]
        result = compute_matches(
            gt, [], pred, [], iou_threshold=0.5, links=links
        )
        assert len(result["tp"]) == 1
        assert result["tp"][0][4] == pytest.approx(4 / 196)

    def test_a_better_overlapping_annotation_keeps_the_prediction(self):
        gt = [(0, 0, 10, 10, 0), (5, 5, 15, 15, 0)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        links = [(("box", 1), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert [t[:4] for t in result["tp"]] == [("box", 0, "box", 0)]
        assert result["fn"] == [("box", 1, 0)]

    def test_the_better_of_two_annotations_linked_to_one_prediction_wins(
        self,
    ):
        gt = [(4, 4, 14, 14, 0), (0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        links = [(("box", 0), ("box", 0)), (("box", 1), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert [t[:4] for t in result["tp"]] == [("box", 1, "box", 0)]
        assert result["fn"] == [("box", 0, 0)]

    def test_a_linked_pair_loses_to_a_better_unlinked_match_elsewhere(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(4, 0, 14, 10, 0, 0.9), (1, 0, 11, 10, 0, 0.9)]
        links = [(("box", 0), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert [t[:4] for t in result["tp"]] == [("box", 0, "box", 1)]
        assert result["fp"] == [("box", 0, 0, 0.9)]

    def test_link_to_a_prediction_under_the_conf_threshold_is_ignored(self):
        gt = [(0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.1)]
        links = [(("box", 0), ("box", 0))]
        result = compute_matches(
            gt, [], pred, [], conf_threshold=0.25, links=links
        )
        assert result["tp"] == [] and result["fp"] == []
        assert len(result["fn"]) == 1

    def test_link_between_shapes_that_do_not_overlap_is_ignored(self):
        gt = [(100, 100, 110, 110, 0)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        links = [(("box", 0), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert result["tp"] == []
        assert len(result["fp"]) == 1 and len(result["fn"]) == 1

    def test_link_between_a_box_and_a_polygon_is_ignored(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = compute_matches(
            [(0, 0, 10, 10, 0)],
            [],
            [],
            [(pts, 0, 0.9)],
            links=[(("box", 0), ("polygon", 0))],
        )
        assert result["tp"] == []

    def test_a_tie_between_linked_annotations_goes_to_the_first(self):
        gt = [(0, 0, 10, 10, 0), (0, 0, 10, 10, 0)]
        pred = [(0, 0, 10, 10, 0, 0.9)]
        links = [(("box", 0), ("box", 0)), (("box", 1), ("box", 0))]
        result = compute_matches(gt, [], pred, [], links=links)
        assert [t[:4] for t in result["tp"]] == [("box", 0, "box", 0)]
        assert result["fn"] == [("box", 1, 0)]
