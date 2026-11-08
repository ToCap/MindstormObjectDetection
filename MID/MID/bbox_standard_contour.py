import cv2


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Initialize the bounding box model.
    Currently stateless (placeholder).
    """
    print("[BBoxModel] initialize called (placeholder)")


def calibrate(ctx=None):
    """
    Optional initialization from a frame.
    """
    print("[BBoxModel] calibrate called (placeholder)")


# ==========================
# DETECTION — BOUNDING BOXES
# ==========================
def detect(
    ctx,
    min_area=200,
    max_area=200000,
    min_aspect=0.1,
    max_aspect=10,
    scale=1.3
):
    edges = ctx.edges

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = edges.shape
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        cx, cy = x + w // 2, y + h // 2
        new_w, new_h = int(w * scale), int(h * scale)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w_img, cx + new_w // 2)
        y2 = min(h_img, cy + new_h // 2)

        boxes.append((x1, y1, x2, y2))

    return boxes


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    boxes,
    color=(0, 255, 0),
    thickness=2
):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )
