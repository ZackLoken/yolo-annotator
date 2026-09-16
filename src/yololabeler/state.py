"""AppState — Single source of truth for all annotation data.
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
        self.document = None         # Document for the current image, or None
        self.load_errors = []        # rejected label-line messages for the current image
        self.verdicts = {}           # live per-image verdict dict from ReviewEngine
        self.current_polygon = []    # in-progress polygon vertices
        self.mode = "polygon"        # "box" | "polygon"

        # Box-drawing temporaries
        self.start_x = None
        self.start_y = None
        self.rect = None

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

        # ── Spatial index (polygon bounding-box cache, by annotation id) ──
        self._poly_bboxes = {}
        self._poly_bboxes_dirty = True

        # ── Annotation interaction ──
        self._dragging_vertex = None       # (annotation id, vertex index)
        self._drag_orig_pos = None
        self._selected_annotation_id = None
        self._hovered_annotation_id = None
        self._stream_mode = False
        self._stream_active = False
        self._last_stream_pos = None

        # ── Review data ──
        self.queue = []              # QueueItem list for the current image
        self.queue_index = 0
        self.predictions = []          # list[Prediction] for the current image
        self.predictions_rejected = [] # "path: line n" messages from loading
        self.predictions_blind = False # True when the image is blind and preds were not read
        self.matches = {}              # compute_matches output for the current image
        self.conf_threshold = 0.50     # mirrored from ReviewEngine.conf_threshold
        self._review_filter_type = "all"
        self._review_filter_class = "all"
        self._review_status_filter = "all"
        self._review_show_pred = True
        self._review_state = {}
        self._annotation_visible = True

        # ── Stats & session tracking ──
        self._stats = {"sessions": [], "image_status": {}, "blind": [], "completion": {}}
        self._current_user = ""
        self._session_start = ""
        self._image_start_time = None
        self._session_annotated_images = set()
        self._session_images = {}
        self._session_loaded_counts = {}
        self._session_add_counts = {}
        self._session_total_adds = 0

        # ── Misc flags ──
        self.show_help = False
        self.banner_text = None
