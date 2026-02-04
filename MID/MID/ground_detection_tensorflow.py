import cv2
import numpy as np
import tensorflow as tf

# ==========================
# GLOBAL MODEL HANDLES
# ==========================
_BACKBONE = None
_SEGMENTATION_HEAD = None


# ==========================
# INITIALIZATION
# ==========================
def init_ground_model(input_size=(224, 224)):
    """
    Initializes the ground segmentation model.
    Must be called ONCE before detection.
    """
    global _BACKBONE, _SEGMENTATION_HEAD

    if _BACKBONE is not None:
        return  # already initialized

    print("[TensorFlow] Initializing ground segmentation model...")

    _BACKBONE = tf.keras.applications.MobileNetV2(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    _BACKBONE.trainable = False

    _SEGMENTATION_HEAD = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
    ])


# ==========================
# IA — MASK PREDICTION
# ==========================
def detect_ground_mask(frame_bgr):
    """
    Predicts a ground probability mask using CNN features.
    """
    if _BACKBONE is None or _SEGMENTATION_HEAD is None:
        raise RuntimeError(
            "Ground model not initialized. "
            "Call init_ground_model() first."
        )

    h, w = frame_bgr.shape[:2]

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img = img_resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    features = _BACKBONE(img, training=False)
    mask = _SEGMENTATION_HEAD(features, training=False)[0, :, :, 0].numpy()

    mask = cv2.resize(mask, (w, h))
    mask = (mask > 0.5).astype(np.uint8)

    return mask


# ==========================
# GEOMETRY — GROUND LINE
# ==========================
def detect_ground_line_from_mask(mask):
    """
    Extracts the ground line as the lowest foreground pixel per column.
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
def draw_ground_mask(overlay, mask, alpha=0.4):
    """
    Draws the ground segmentation mask.
    """
    color = np.zeros_like(overlay)
    color[mask > 0] = (255, 0, 0)
    cv2.addWeighted(color, alpha, overlay, 1 - alpha, 0, overlay)


def draw_ground_line(overlay, y):
    """
    Draws the detected ground line.
    """
    if y is None:
        return

    h, w = overlay.shape[:2]

    cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 4)
    cv2.putText(
        overlay,
        "GROUND (AI)",
        (20, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )
