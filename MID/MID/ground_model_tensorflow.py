import cv2
import numpy as np
import tensorflow as tf


# ==========================
# GLOBAL MODEL STATE
# ==========================
_GROUND_BACKBONE = None
_GROUND_HEAD = None
_GROUND_INPUT_SIZE = (224, 224)


# ==========================
# INITIALIZATION
# ==========================
def init_ground_model(input_size=(224, 224)):
    """
    Initializes the TensorFlow ground segmentation model.
    Must be called once before any detection.
    """
    global _GROUND_BACKBONE, _GROUND_HEAD, _GROUND_INPUT_SIZE

    if _GROUND_BACKBONE is not None:
        return  # already initialized

    print("[TensorFlow] Initializing ground model...")

    _GROUND_INPUT_SIZE = input_size

    _GROUND_BACKBONE = tf.keras.applications.MobileNetV2(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    _GROUND_BACKBONE.trainable = False

    _GROUND_HEAD = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
    ])


def is_ground_model_initialized():
    """
    Returns True if the ground model is initialized.
    """
    return _GROUND_BACKBONE is not None and _GROUND_HEAD is not None


# ==========================
# DETECTION — GROUND MASK
# ==========================
def detect_ground_mask(frame_bgr, threshold=0.5):
    """
    Predicts a binary ground mask from a BGR frame.
    """
    if not is_ground_model_initialized():
        raise RuntimeError(
            "Ground model not initialized. "
            "Call init_ground_model() first."
        )

    h, w = frame_bgr.shape[:2]

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, _GROUND_INPUT_SIZE)
    img = img_resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    features = _GROUND_BACKBONE(img, training=False)
    mask = _GROUND_HEAD(features, training=False)[0, :, :, 0].numpy()

    mask = cv2.resize(mask, (w, h))
    mask = (mask > threshold).astype(np.uint8)

    return mask


# ==========================
# GEOMETRY — GROUND LINE
# ==========================
def detect_ground_line(mask):
    """
    Extracts the ground line as the median of the lowest
    detected ground pixel per image column.
    """
    h, w = mask.shape
    y_candidates = []

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            y_candidates.append(ys[-1])

    if not y_candidates:
        return None

    return int(np.median(y_candidates))


# ==========================
# DRAWING
# ==========================
def draw_ground_mask(
    overlay,
    mask,
    color=(255, 0, 0),
    alpha=0.4
):
    """
    Draws the ground segmentation mask.
    """
    color_layer = np.zeros_like(overlay)
    color_layer[mask > 0] = color

    cv2.addWeighted(
        color_layer,
        alpha,
        overlay,
        1 - alpha,
        0,
        overlay
    )


def draw_ground_line(
    overlay,
    y,
    color=(255, 0, 0),
    thickness=4
):
    """
    Draws the detected ground line.
    """
    if y is None:
        return

    h, w = overlay.shape[:2]

    cv2.line(overlay, (0, y), (w, y), color, thickness)
    cv2.putText(
        overlay,
        "GROUND",
        (20, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )
