"""AppState — Single source of truth for all annotation data.

This class is GUI-free and can be instantiated headlessly for
programmatic access (AI agents, training pipelines, CLI tools).
"""


class AppState:
    """Owns all annotation and review data.  No GUI, no rendering, no I/O."""

    def __init__(self):
        # ── Paths ──
        self.image_folder = ""
        self.labels_dir = ""
        self.detect_dir = ""
        self.segment_dir = ""
        self.state_dir = ""
        self.pred_detect_dir = None
        self.pred_segment_dir = None

        # ── Image list & current image ──
        self.images = []
        self.index = 0
        self.original_image = None   # PIL Image or None
        self.img_width = 0
        self.img_height = 0

        # ── Annotation data ──
        self.boxes = []              # [(class_id, x1, y1, x2, y2), ...]
        self.polygons = []           # [(class_id, [x1,y1, ...]), ...]
        self.current_polygon = []    # in-progress polygon vertices
        self.mode = "polygon"        # "box" | "polygon"

        # Box-drawing temporaries
        self.start_x = None
        self.start_y = None
        self.rect = None

        # ── Predictions (read-only, for review tab) ──
        self.pred_boxes = []
        self.pred_polygons = []

        # ── Class registry ──
        self.class_names = {}
        self.class_colors = {}
        self.active_class = 0

        # ── Undo / redo ──
        self._undo_stack = []
        self._redo_stack = []
        self._vertex_redo_stack = []

        # ── Snap ──
        self.snap_enabled = False

        # ── Completion / filter ──
        self._completed_images = set()
        self._active_filter = "all"
        self._filtered_indices = []

        # ── Spatial index (polygon bounding-box cache) ──
        self._poly_bboxes = []
        self._poly_bboxes_dirty = True

        # ── Annotation interaction ──
        self._dragging_vertex = None
        self._drag_orig_pos = None
        self._selected_polygon_idx = None
        self._hovered_polygon_idx = None
        self._stream_mode = False
        self._stream_active = False
        self._last_stream_pos = None

        # ── Review data ──
        self._review_index = 0
        self._review_detection_idx = 0
        self._review_detections = []
        self._review_matches = {}
        self._review_gt_boxes = []
        self._review_gt_polygons = []
        self._review_pred_boxes = []
        self._review_pred_polygons = []
        self._review_original_image = None
        self._review_img_w = 0
        self._review_img_h = 0
        self._review_scale = 1.0
        self._review_offset_x = 0.0
        self._review_offset_y = 0.0
        self._review_cached_scale = None
        self._review_filter_type = "all"
        self._review_filter_class = "all"
        self._review_pan_start_x = None
        self._review_pan_start_y = None
        self._review_state = {}
        self._reviewed_lookup = ("", {}, {})
        self._review_show_gt = True
        self._review_show_pred = True
        self._review_filtered_images = []
        self._review_status_filter = "all"
        self._review_needs_first_zoom = False
        self._review_det_reviewed = {}
        self._review_show_help = False
        self._annotation_visible = True
        self._annotate_pred_reference = None
        self._review_return_pending = False
        self._review_editing_det = None
        self._review_recompute_on_return = False

        # ── Stats & session tracking ──
        self._stats = {"sessions": [], "image_status": {}}
        self._image_dims = {}
        self._current_user = ""
        self._session_start = ""
        self._image_start_time = None
        self._review_image_start_time = None
        self._session_annotated_images = set()
        self._session_images = {}
        self._session_loaded_counts = {}
        self._session_add_counts = {}
        self._session_total_adds = 0

        # ── Misc flags ──
        self._defer_display = False
        self.show_help = False
