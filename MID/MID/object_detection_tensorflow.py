import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# ==========================
# COCO LABELS
# ==========================
LABELS = [
    "???", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror",
    "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


# ==========================
# LOAD MODEL (ONCE)
# ==========================
print("[TensorFlow] Loading object detection model...")
MODEL = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")


# ==========================
# OBJECT DETECTION
# ==========================
def detect_objects(frame_bgr):
    """
    Runs TensorFlow object detection on a BGR frame.

    Returns:
    - boxes   : [N, 4] in normalized coordinates
    - scores  : [N]
    - classes : [N]
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    image_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)

    outputs = MODEL(image_tensor)

    boxes = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy()

    return boxes, scores, classes


# ==========================
# DRAW OBJECT BOUNDING BOXES
# ==========================
def draw_object_boxes(
    frame,
    boxes,
    scores,
    classes,
    score_threshold=0.5
):
    """
    Draws object bounding boxes detected by TensorFlow.
    """
    h, w, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        class_id = int(classes[i])
        label = LABELS[class_id] if class_id < len(LABELS) else f"class {class_id}"
        text = f"{label} ({scores[i] * 100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

