import json
import os
from datetime import datetime
from enum import IntEnum

import cv2
import numpy as np

import bbox_standard_contour as bbox_contour
import ground_standard_texture as ground_tex


# ==========================
# CONFIGURATION
# ==========================
# MAP_MODE selects the geometry of the grid:
#  - "PIXEL": cells are blocks of CELL_SIZE_PX x CELL_SIZE_PX pixels of the
#    camera image (no calibration needed, no distances in meters)
#  - "METRIC": cells are CELL_SIZE_M squares on the ground around the robot
#    (needs the pixel-to-meter calibration)
MAP_MODE = "PIXEL"
CELL_SIZE_PX = 16           # PIXEL mode: side of one cell, in pixels
GROUND_COVERAGE = 0.5       # PIXEL mode: share of ground pixels for a ground cell

GRID_WIDTH_M = 4.0          # grid width, in meters
GRID_DEPTH_M = 4.0          # grid depth in front of the robot, in meters
CELL_SIZE_M = 0.05          # side of one cell, in meters
OBSTACLE_MARGIN_CELLS = 1   # thickness added around the contact segment of a box
BBOX_SCALE = 1.0            # enlargement requested from bbox_contour (1.0 = none)

CALIB_RECT_WIDTH_M = 0.40   # width of the rectangle laid on the floor
CALIB_RECT_DEPTH_M = 0.60   # depth of that rectangle
CALIB_RECT_NEAR_M = 0.30    # distance between the robot and the near edge
CALIB_FILE = "map_calibration.json"

DRAW_GRID_STEP_M = 0.5      # spacing of the ticks and of the control grid

# Sources fused into the grid, as (module, keyword arguments of detect).
# Another model is added by appending it to the right list.
#  - ground source: detect(ctx) -> uint8 image of the frame size, 0 = no
#    information, 1 = reference ground surface, >= 2 = ground of another
#    surface (both are free, no obstacle); calibrate(ctx) / is_calibrated()
#    are used if the module has them
#  - obstacle source: detect(ctx) -> list of (x1, y1, x2, y2) boxes whose
#    bottom edge touches the ground
GROUND_SOURCES = [(ground_tex, {})]
OBSTACLE_SOURCES = [(bbox_contour, {"scale": BBOX_SCALE})]


# interactive calibration: main.py asks for it only once, not at every frame
CALIBRATION_ASKED_ONCE = True


# ==========================
# TYPES AND ERROR CODES
# ==========================
class CellState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    FREE_OTHER_SURFACE = 3   # free, but another surface than the reference


MAP_E_NOT_INITIALIZED = 0x01
MAP_E_NOT_CALIBRATED = 0x02
MAP_E_CALIB_ABORTED = 0x03
MAP_E_CALIB_FILE = 0x04
MAP_E_BAD_FRAME = 0x05
MAP_E_WRONG_MODE = 0x06

_ERROR_NAMES = {
    MAP_E_NOT_INITIALIZED: "MAP_E_NOT_INITIALIZED",
    MAP_E_NOT_CALIBRATED: "MAP_E_NOT_CALIBRATED",
    MAP_E_CALIB_ABORTED: "MAP_E_CALIB_ABORTED",
    MAP_E_CALIB_FILE: "MAP_E_CALIB_FILE",
    MAP_E_BAD_FRAME: "MAP_E_BAD_FRAME",
    MAP_E_WRONG_MODE: "MAP_E_WRONG_MODE",
}


# ==========================
# GLOBAL MODEL STATE
# ==========================
_initialized = False
_N_ROWS = 0
_N_COLS = 0

_H = None            # image pixels -> ground meters (robot frame)
_H_INV = None        # ground meters -> image pixels
_H_GRID = None       # image pixels -> grid cells (column, row)
_IMAGE_SIZE = None   # (width, height) of the calibrated image
_IMAGE_POINTS = None
_GROUND_POINTS = None
_FROM_FILE = False   # True if the calibration was reloaded from CALIB_FILE
_WARNED_NOT_CALIBRATED = False

_CAMERA_INDEX = 1    # only used when calibrate() is called without a context
_WINDOW = "Map calibration"
_POINT_LABELS = ["near left", "near right", "far right", "far left"]


# ==========================
# INTERNAL HELPERS
# ==========================
def _report(code, message):
    print("[MapFusion] %s (0x%02X): %s" % (_ERROR_NAMES[code], code, message))


def _require_initialized():
    if not _initialized:
        raise RuntimeError(
            "[MapFusion] %s (0x%02X): call initialize() first"
            % (_ERROR_NAMES[MAP_E_NOT_INITIALIZED], MAP_E_NOT_INITIALIZED)
        )


def _check_frame(frame):
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError(
            "[MapFusion] %s (0x%02X): empty frame"
            % (_ERROR_NAMES[MAP_E_BAD_FRAME], MAP_E_BAD_FRAME)
        )


def _transform(matrix, points):
    """Applies a 3x3 perspective matrix to a list of 2D points (N x 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _scaling_matrix():
    """Meters (x right, y forward) -> cell indices (column, row)."""
    s = CELL_SIZE_M
    return np.array([
        [1.0 / s, 0.0, GRID_WIDTH_M / (2.0 * s)],
        [0.0, -1.0 / s, GRID_DEPTH_M / s],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _meters_to_cells(points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cols = (pts[:, 0] + GRID_WIDTH_M / 2.0) / CELL_SIZE_M
    rows = (GRID_DEPTH_M - pts[:, 1]) / CELL_SIZE_M
    return np.stack([cols, rows], axis=1)


def _to_int_points(points):
    pts = np.nan_to_num(np.asarray(points, dtype=np.float64), nan=0.0)
    return np.round(np.clip(pts, -1e6, 1e6)).astype(np.int32)


def _calibration_ground_points():
    w = CALIB_RECT_WIDTH_M
    length = CALIB_RECT_DEPTH_M
    d = CALIB_RECT_NEAR_M
    return [(-w / 2, d), (w / 2, d), (w / 2, d + length), (-w / 2, d + length)]


def _set_calibration(h_matrix, image_size, image_points, ground_points,
                     from_file):
    global _H, _H_INV, _H_GRID, _IMAGE_SIZE
    global _IMAGE_POINTS, _GROUND_POINTS, _FROM_FILE

    h_matrix = np.asarray(h_matrix, dtype=np.float64).reshape(3, 3)
    _H = h_matrix
    _H_INV = np.linalg.inv(h_matrix)
    _H_GRID = _scaling_matrix() @ h_matrix
    _IMAGE_SIZE = (int(image_size[0]), int(image_size[1]))
    _IMAGE_POINTS = [(int(p[0]), int(p[1])) for p in image_points]
    _GROUND_POINTS = [(float(p[0]), float(p[1])) for p in ground_points]
    _FROM_FILE = from_file


def _clear_calibration():
    global _H, _H_INV, _H_GRID, _IMAGE_SIZE
    global _IMAGE_POINTS, _GROUND_POINTS, _FROM_FILE

    _H = None
    _H_INV = None
    _H_GRID = None
    _IMAGE_SIZE = None
    _IMAGE_POINTS = None
    _GROUND_POINTS = None
    _FROM_FILE = False


def _load_calibration():
    if not os.path.isfile(CALIB_FILE):
        return

    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)

        _set_calibration(
            data["H"], data["image_size"],
            data["image_points"], data["ground_points"], True
        )
        print("[MapFusion] Calibration loaded from %s" % CALIB_FILE)

    except (OSError, ValueError, KeyError, TypeError, IndexError,
            np.linalg.LinAlgError) as exc:
        _clear_calibration()
        _report(MAP_E_CALIB_FILE, "%s ignored (%s)" % (CALIB_FILE, exc))


def _save_calibration():
    data = {
        "image_size": list(_IMAGE_SIZE),
        "image_points": [list(p) for p in _IMAGE_POINTS],
        "ground_points": [list(p) for p in _GROUND_POINTS],
        "H": _H.tolist(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("[MapFusion] Calibration saved in %s" % CALIB_FILE)


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Initializes the sources and, in METRIC mode, computes the grid size and
    reloads the calibration file if it exists (in PIXEL mode the grid size
    follows the frame size).
    Must be called once before any other function.
    """
    global _initialized, _N_ROWS, _N_COLS, _WARNED_NOT_CALIBRATED

    if MAP_MODE not in ("PIXEL", "METRIC"):
        raise ValueError("[MapFusion] unknown MAP_MODE %r" % (MAP_MODE,))

    for module, _ in GROUND_SOURCES + OBSTACLE_SOURCES:
        module.initialize()

    _WARNED_NOT_CALIBRATED = False
    _clear_calibration()

    if MAP_MODE == "METRIC":
        _N_COLS = int(round(GRID_WIDTH_M / CELL_SIZE_M))
        _N_ROWS = int(round(GRID_DEPTH_M / CELL_SIZE_M))
        _load_calibration()
        print("[MapFusion] Initialized, METRIC mode (%d x %d cells)"
              % (_N_ROWS, _N_COLS))
    else:
        _N_COLS = 0
        _N_ROWS = 0
        print("[MapFusion] Initialized, PIXEL mode (%d px cells)"
              % CELL_SIZE_PX)

    _initialized = True


# ==========================
# PIXEL <-> METER EQUIVALENCE
# ==========================
def is_calibrated():
    """
    True if the map can be built: always in PIXEL mode (nothing to
    calibrate), in METRIC mode once the homography is available.
    """
    return MAP_MODE == "PIXEL" or _H is not None


def pixel_to_ground(u, v):
    """
    Ground position (x right, y forward, in meters) of the pixel (u, v).
    Valid for a pixel that touches the ground. None if not calibrated.
    """
    if _H is None:
        return None

    x, y = _transform(_H, [(u, v)])[0]
    return float(x), float(y)


def ground_to_pixel(x, y):
    """Pixel (u, v) of the ground point (x, y). None if not calibrated."""
    if _H_INV is None:
        return None

    u, v = _transform(_H_INV, [(x, y)])[0]
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def ground_resolution(u, v):
    """
    Size on the ground of one pixel at (u, v), as (dx, dy) in meters per
    pixel. None if not calibrated.
    """
    if _H is None:
        return None

    p0 = pixel_to_ground(u, v)
    px = pixel_to_ground(u + 1, v)
    py = pixel_to_ground(u, v + 1)
    return abs(px[0] - p0[0]), abs(py[1] - p0[1])


# ==========================
# CALIBRATION (interactive)
# ==========================
def _put(view, text, org, color=(255, 255, 255)):
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


def _select_points(frame):
    """
    Lets the operator click the 4 corners of the rectangle (and optionally
    its centre as a 5th point). Returns the points, or None if cancelled.
    """
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
            points.append((x, y))

    cv2.namedWindow(_WINDOW)
    cv2.setMouseCallback(_WINDOW, on_mouse)

    while True:
        view = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(view, p, 4, (0, 0, 255), -1)
            _put(view, str(i + 1), (p[0] + 6, p[1] - 6), (0, 255, 255))
        if len(points) >= 2:
            quad = np.array(points[:4], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [quad], len(points) >= 4, (0, 255, 255), 1)

        if len(points) < 4:
            _put(view, "Click corner %d/4: %s"
                 % (len(points) + 1, _POINT_LABELS[len(points)]), (10, 20))
        elif len(points) == 4:
            _put(view, "Enter: validate | optional 5th click: rectangle "
                 "centre (check)", (10, 20))
        else:
            _put(view, "Enter: validate", (10, 20))
        _put(view, "r: restart | Esc: cancel", (10, 40))

        cv2.imshow(_WINDOW, view)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            return None
        if key == ord("r"):
            points.clear()
        if key == 13 and len(points) >= 4:
            return list(points)


def _polyline(view, points, color):
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return
    cv2.polylines(view, [_to_int_points(pts).reshape(-1, 1, 2)],
                  False, color, 1, cv2.LINE_AA)


def _draw_control_grid(view, h_inv):
    """Overlays a metric grid (one line every DRAW_GRID_STEP_M)."""
    step = DRAW_GRID_STEP_M
    half = GRID_WIDTH_M / 2.0
    xs = np.arange(-half, half + 1e-9, step)
    ys = np.arange(step, GRID_DEPTH_M + 1e-9, step)
    h_img, w_img = view.shape[:2]

    for x in xs:
        line = [(x, y) for y in np.linspace(ys[0], GRID_DEPTH_M, 20)]
        _polyline(view, _transform(h_inv, line), (0, 255, 0))

    for y in ys:
        line = [(x, y) for x in np.linspace(-half, half, 20)]
        pts = _transform(h_inv, line)
        _polyline(view, pts, (0, 255, 0))

        u, v = pts[0]
        if np.isfinite(u) and np.isfinite(v) and 0 <= u < w_img and 0 <= v < h_img:
            _put(view, "%.1f m" % y, (int(u) + 4, int(v) - 4), (0, 255, 0))


def _review(frame, h_matrix, points):
    """
    Shows the metric control grid and the ground resolution.
    Returns "accept", "redo" or "cancel".
    """
    h_inv = np.linalg.inv(h_matrix)
    view = frame.copy()
    _draw_control_grid(view, h_inv)

    for p in points[:4]:
        cv2.circle(view, p, 4, (0, 0, 255), -1)

    lines = []

    if len(points) == 5:
        expected = (0.0, CALIB_RECT_NEAR_M + CALIB_RECT_DEPTH_M / 2.0)
        x, y = _transform(h_matrix, [points[4]])[0]
        error_cm = 100.0 * float(np.hypot(x - expected[0], y - expected[1]))
        lines.append("Position error at the centre: %.1f cm" % error_cm)

    # ground resolution at 1 m, 2 m and 3 m (only inside the grid)
    for d in (1.0, 2.0, 3.0):
        if d > GRID_DEPTH_M:
            continue
        u, v = _transform(h_inv, [(0.0, d)])[0]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        p0 = _transform(h_matrix, [(u, v)])[0]
        px = _transform(h_matrix, [(u + 1, v)])[0]
        py = _transform(h_matrix, [(u, v + 1)])[0]
        dx_cm = 100.0 * abs(px[0] - p0[0])
        dy_cm = 100.0 * abs(py[1] - p0[1])
        lines.append("%.0f m: %.2f cm/px (x), %.2f cm/px (y)" % (d, dx_cm, dy_cm))

    for i, text in enumerate(lines):
        print("[MapFusion] " + text)
        _put(view, text, (10, 20 + 20 * i))

    _put(view, "Enter: accept and save | r: redo | Esc: cancel",
         (10, view.shape[0] - 12))

    while True:
        cv2.imshow(_WINDOW, view)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            return "cancel"
        if key == ord("r"):
            return "redo"
        if key == 13:
            return "accept"


def calibrate(ctx=None):
    """
    Interactive pixel-to-meter calibration: the operator clicks the 4
    corners of a rectangle of known size laid on the floor.
    Returns True if validated and saved, False if cancelled.
    """
    _require_initialized()

    if MAP_MODE != "METRIC":
        _report(MAP_E_WRONG_MODE, "calibrate() is only available in METRIC mode")
        return False

    if ctx is not None:
        frame = ctx.frame.copy()
    else:
        cap = cv2.VideoCapture(_CAMERA_INDEX)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None

    _check_frame(frame)
    h_img, w_img = frame.shape[:2]
    ground_points = _calibration_ground_points()

    while True:
        points = _select_points(frame)
        if points is None:
            decision = "cancel"
        else:
            h_matrix = cv2.getPerspectiveTransform(
                np.float32(points[:4]), np.float32(ground_points)
            )
            decision = _review(frame, h_matrix, points)

        if decision == "redo":
            continue

        cv2.destroyWindow(_WINDOW)

        if decision == "cancel":
            _report(MAP_E_CALIB_ABORTED, "calibration cancelled")
            return False

        _set_calibration(
            h_matrix, (w_img, h_img), points[:4], ground_points, False
        )
        _save_calibration()
        return True


# ==========================
# DETECTION - OCCUPANCY GRID
# ==========================
def _project_box(box, obstacle, shadow):
    """
    Draws the contact segment of a box in 'obstacle' and the region hidden
    behind it in 'shadow' (both in grid cells).
    """
    x1, y1, x2, y2 = box
    segment_px = [(x1, y2), (x2, y2)]

    cells = _transform(_H_GRID, segment_px)
    ground = _transform(_H, segment_px)
    if not (np.all(np.isfinite(cells)) and np.all(np.isfinite(ground))):
        return

    # a box whose segment lies entirely outside the grid is ignored
    p1, p2 = (tuple(int(v) for v in p) for p in _to_int_points(cells))
    inside, _, _ = cv2.clipLine((0, 0, _N_COLS, _N_ROWS), p1, p2)
    if not inside:
        return

    cv2.line(obstacle, p1, p2, 1, 1 + 2 * OBSTACLE_MARGIN_CELLS)

    # shadow: the two segment ends extended along the rays from the robot
    if np.all(ground[:, 1] > 0):
        reach = 2.0 * np.hypot(GRID_WIDTH_M, GRID_DEPTH_M)
        far = ground * (reach / np.hypot(ground[:, 0], ground[:, 1]))[:, None]
        polygon = np.array([ground[0], ground[1], far[1], far[0]])
        cv2.fillPoly(shadow, [_to_int_points(_meters_to_cells(polygon))], 1)


def _warn_not_calibrated():
    global _WARNED_NOT_CALIBRATED

    if not _WARNED_NOT_CALIBRATED:
        _report(MAP_E_NOT_CALIBRATED, "no pixel-to-meter calibration, "
                "the grid is entirely UNKNOWN")
        _WARNED_NOT_CALIBRATED = True


def _merge_surface(surface, cls):
    """Merges a surface map: the reference surface (1) wins over 2."""
    return np.where(
        cls > 0, np.where(surface > 0, np.minimum(surface, cls), cls), surface
    ).astype(np.uint8)


def _mask_to_cells_pixel(mask):
    """
    PIXEL mode: surface of every cell from a ground mask of the image size
    (0 = none, 1 = reference, 2 = other). A cell is ground if at least
    GROUND_COVERAGE of its pixels are, with the majority surface.
    """
    s = CELL_SIZE_PX
    h, w = mask.shape[:2]
    cls = np.minimum(mask, 2)

    share = []
    for k in (1, 2):
        part = cv2.copyMakeBorder(
            (cls == k).astype(np.float32), 0, _N_ROWS * s - h,
            0, _N_COLS * s - w, cv2.BORDER_REPLICATE
        )
        share.append(cv2.resize(part, (_N_COLS, _N_ROWS),
                                interpolation=cv2.INTER_AREA))

    covered = (share[0] + share[1]) >= GROUND_COVERAGE
    cells = np.where(share[0] >= share[1], 1, 2).astype(np.uint8)
    cells[~covered] = 0
    return cells


def _project_box_pixel(box, obstacle, shadow):
    """
    PIXEL mode: the bottom edge of the box gives the obstacle cells, and
    the cells above it (behind it in the image) are the shadow.
    """
    s = CELL_SIZE_PX
    x1, y1, x2, y2 = box

    # a box entirely outside the grid is ignored
    if x2 < 0 or y2 < 0 or x1 >= _N_COLS * s:
        return

    c1 = int(np.clip(x1 // s, 0, _N_COLS - 1))
    c2 = int(np.clip(x2 // s, 0, _N_COLS - 1))
    row = int(np.clip(y2 // s, 0, _N_ROWS - 1))
    m = OBSTACLE_MARGIN_CELLS

    obstacle[max(row - m, 0):min(row + m, _N_ROWS - 1) + 1,
             max(c1 - m, 0):min(c2 + m, _N_COLS - 1) + 1] = 1
    shadow[:max(row - m, 0), c1:c2 + 1] = 1


def detect(ctx):
    """
    Builds the occupancy grid of the current frame.
    Priority per cell: OBSTACLE > hidden behind an obstacle (UNKNOWN)
    > FREE / FREE_OTHER_SURFACE (ground) > UNKNOWN.
    """
    global _N_ROWS, _N_COLS

    _require_initialized()

    frame = ctx.frame
    _check_frame(frame)
    h_img, w_img = frame.shape[:2]

    if MAP_MODE == "PIXEL":
        _N_COLS = -(-w_img // CELL_SIZE_PX)
        _N_ROWS = -(-h_img // CELL_SIZE_PX)
        project_box = _project_box_pixel
    else:
        project_box = _project_box

    grid = np.full((_N_ROWS, _N_COLS), int(CellState.UNKNOWN), dtype=np.uint8)

    if MAP_MODE == "METRIC":
        if _H is None:
            _warn_not_calibrated()
            return grid

        if (w_img, h_img) != _IMAGE_SIZE:
            if _FROM_FILE:
                _report(MAP_E_CALIB_FILE,
                        "%s was made for %dx%d images, ignored"
                        % (CALIB_FILE, _IMAGE_SIZE[0], _IMAGE_SIZE[1]))
                _clear_calibration()
                return grid
            raise ValueError(
                "[MapFusion] %s (0x%02X): frame is %dx%d, calibration is %dx%d"
                % (_ERROR_NAMES[MAP_E_BAD_FRAME], MAP_E_BAD_FRAME,
                   w_img, h_img, _IMAGE_SIZE[0], _IMAGE_SIZE[1])
            )

    # ground sources: reference surface (1) wins over another surface (2)
    surface = np.zeros((_N_ROWS, _N_COLS), dtype=np.uint8)
    for module, kwargs in GROUND_SOURCES:
        if hasattr(module, "is_calibrated") and not module.is_calibrated():
            module.calibrate(ctx)

        mask = module.detect(ctx, **kwargs)
        if MAP_MODE == "PIXEL":
            cls = _mask_to_cells_pixel(mask)
        else:
            cls = np.minimum(cv2.warpPerspective(
                mask, _H_GRID, (_N_COLS, _N_ROWS),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ), 2)
        surface = _merge_surface(surface, cls)

    # obstacle sources: contact segment and shadow of every box
    obstacle = np.zeros((_N_ROWS, _N_COLS), dtype=np.uint8)
    shadow = np.zeros((_N_ROWS, _N_COLS), dtype=np.uint8)
    for module, kwargs in OBSTACLE_SOURCES:
        for box in module.detect(ctx, **kwargs):
            project_box(box, obstacle, shadow)

    grid[surface == 1] = CellState.FREE
    grid[surface == 2] = CellState.FREE_OTHER_SURFACE
    grid[shadow > 0] = CellState.UNKNOWN
    grid[obstacle > 0] = CellState.OBSTACLE

    return grid


# ==========================
# DRAWING
# ==========================
_PALETTE = np.array([
    (128, 128, 128),   # UNKNOWN: gray
    (0, 200, 0),       # FREE: green
    (0, 0, 255),       # OBSTACLE: red
    (200, 200, 0),     # FREE_OTHER_SURFACE: cyan
], dtype=np.uint8)


def draw(overlay, grid):
    """
    Replaces the content of the view by the grid, scaled to the view
    (nearest neighbour, aspect ratio kept, black bands).
    """
    _require_initialized()

    if grid.shape != (_N_ROWS, _N_COLS):
        raise ValueError("[MapFusion] unexpected grid shape %s" % (grid.shape,))

    h_view, w_view = overlay.shape[:2]
    scale = min(w_view / _N_COLS, h_view / _N_ROWS)
    w_grid = max(1, int(_N_COLS * scale))
    h_grid = max(1, int(_N_ROWS * scale))
    x0 = (w_view - w_grid) // 2
    y0 = (h_view - h_grid) // 2

    canvas = np.zeros_like(overlay)
    canvas[y0:y0 + h_grid, x0:x0 + w_grid] = cv2.resize(
        _PALETTE[grid], (w_grid, h_grid), interpolation=cv2.INTER_NEAREST
    )

    # METRIC mode: ticks every DRAW_GRID_STEP_M (distance and lateral lines);
    # in PIXEL mode there is no distance to show
    if MAP_MODE == "METRIC":
        px_per_m = w_grid / GRID_WIDTH_M
        tick = (210, 210, 210)

        for y in np.arange(DRAW_GRID_STEP_M, GRID_DEPTH_M + 1e-9,
                           DRAW_GRID_STEP_M):
            yy = y0 + h_grid - int(round(y * px_per_m))
            cv2.line(canvas, (x0, yy), (x0 + w_grid - 1, yy), tick, 1)
            cv2.putText(canvas, "%.1f m" % y, (x0 + 3, yy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)

        half = GRID_WIDTH_M / 2.0
        for x in np.arange(-half + DRAW_GRID_STEP_M, half - 1e-9,
                           DRAW_GRID_STEP_M):
            xx = x0 + int(round((x + half) * px_per_m))
            cv2.line(canvas, (xx, y0), (xx, y0 + h_grid - 1), tick, 1)

    # robot: bottom centre
    cx = x0 + w_grid // 2
    by = y0 + h_grid - 1
    robot = np.array([(cx, by - 14), (cx - 8, by), (cx + 8, by)], np.int32)
    cv2.fillPoly(canvas, [robot], (255, 255, 255))

    overlay[:] = canvas
