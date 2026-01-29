import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# COCO label list
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


def detect_objects(image_np):
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)  # add batch dimension
    outputs = model(image_tensor)
    return {k: v.numpy() for k, v in outputs.items()}


def dddd(circles):

    filtered = []
    for c in circles:
        x1, y1, r1 = c
        keep_circle = True

        for i, fc in enumerate(filtered):
            x2, y2, r2 = fc

            # Calcul de distance entre centres
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Si les centres sont trop proches
            if dist < min_dist:
                # Garder le cercle avec le plus grand rayon
                if r1 > r2:
                    filtered[i] = c     # remplacer l'ancien
                keep_circle = False       # ne pas ajouter le cercle actuel
                break

        if keep_circle:
            filtered.append(c)






print("Loading model...")

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")


# open intenal camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error : unable to open camera")
# HINTS : index corresponds to the camera in the device lists


while True:

    # open stream from camera
    ret, frame = cap.read()
    if not ret:
        print("Error : Unable to read frame from camera")
        break

    # copy frame for processing
    raw_frame = frame.copy()
    overlay = frame.copy()

    # compute approx position of camera
    h, w = frame.shape[:2]
    cam_x = w // 2       # centre horizontal
    cam_y = h - 1        # dernier pixel en bas
    
    # convert frame in gray-scale for faster processing
    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)#conversion to gray-scale
    #gray = cv2.GaussianBlur(gray, (5,5), 0)#apply Gaussian blur

    #overlay = gray

    #edges = cv2.Canny(gray, 100, 200)
    edges = cv2.Canny(gray, 30, 100)
    #edges = cv2.Canny(gray, lower, upper)

    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    # 1) Créer un kernel 5x5 (matrice de 1)
    kernel = np.ones((3, 3), np.uint8)

    # 2) Fermer les petits trous dans les contours (dilatation puis érosion)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3) Enlever les petits points parasites (érosion puis dilatation)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)

    #
    if 0:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # vert

    # 
    if 0:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(overlay, (x, y), r, (255, 0, 0), 2)  # bleu
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), 3)  # centre

    # 
    if 0:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(overlay, [approx], -1, (0, 0, 255), 2)  # rouge

            # find the lowest horiztal line
            pts = approx.reshape(-1, 2)
            max_y = np.max(pts[:, 1])

            # LIGNE DE DISTANCE
            #cv2.circle(overlay, (cam_x, cam_y), 5, (255, 0, 0),-1)
            #cv2.line(overlay,(cam_x, cam_y),(cam_x, max_y),(0, 255, 0),2)

    if 0:

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = overlay.shape[:2]

        for cnt in contours:

            area = cv2.contourArea(cnt)

            ## --- FILTRAGE ---
            #if area < 500:      # trop petit → bruit
            #    continue
            #if area > 50000:    # trop grand → fond
            #    continue

            if area < 200:   # au lieu de 500
                continue
            if area > 200000:
                continue

            # Rectangle englobant original
            x, y, w, h = cv2.boundingRect(cnt)

            # Ratio largeur / hauteur (évite lignes parasites)
            aspect_ratio = w / float(h)
            #if aspect_ratio < 0.2 or aspect_ratio > 5:
            #    continue
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue

            # -------- MÉTHODE 3 : AGRANDIR DEPUIS LE CENTRE --------
            scale = 1.3  # 30% plus grand (ajuste si besoin)

            cx = x + w // 2
            cy = y + h // 2

            new_w = int(w * scale)
            new_h = int(h * scale)

            x2 = max(0, cx - new_w // 2)
            y2 = max(0, cy - new_h // 2)
            x3 = min(w_img, cx + new_w // 2)
            y3 = min(h_img, cy + new_h // 2)

            # Dessiner bounding box agrandie
            cv2.rectangle(
                overlay,
                (x2, y2),
                (x3, y3),
                (0, 255, 0),
                2
            )




    if 0:
        # convert BGR to RGB as TensorFlow expects RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # start detection
        image_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
        image_tensor = tf.expand_dims(image_tensor, 0)  # add batch dimension
        outputs = model(image_tensor)
        results = {k: v.numpy() for k, v in outputs.items()}

 
        # resize output arrays to remove batch from tensorflow 
        boxes   = np.squeeze(results["detection_boxes"])
        scores  = np.squeeze(results["detection_scores"])
        classes = np.squeeze(results["detection_classes"])
        h, w, _ = frame.shape

        # build bounding box for each detected object 
        for i in range(len(scores)):
        
            # ignore results with score lower than 80%
            score = scores[i]
            if score < 0.5:
                continue

            # compute coordinates from tensorflow results
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)

            # 
            class_id = int(classes[i])
            label = LABELS[class_id] if class_id < len(LABELS) else f"class {class_id}"
            text = f"{label} ({score*100:.1f}%)"

            # create bounding box into current frame
            color = (0, 255, 0) # color depends on confidence score
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    if 1:
         # ==========================
        # DÉTECTION + TRACE DU SOL
        # ==========================

        h_img, w_img = edges.shape

        # Zone de recherche : bas de l'image
        roi_y = int(h_img * 0.6)
        edges_roi = edges[roi_y:h_img, :]

        lines = cv2.HoughLinesP(
            edges_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=80,                 # ↓ plus permissif
            minLineLength=int(w_img * 0.3),
            maxLineGap=50
        )

        best_line = None
        best_score = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Recaler dans l'image complète
                y1 += roi_y
                y2 += roi_y

                dy = abs(y1 - y2)
                dx = abs(x2 - x1)

                # quasi horizontal
                if dy < 15 and dx > 50:
                    score = dx - dy * 5  # favorise longues lignes plates

                    if score > best_score:
                        best_score = score
                        best_line = (x1, y1, x2, y2)

        # TRACE DU SOL
        if best_line is not None:
            x1, y1, x2, y2 = best_line
            y = (y1 + y2) // 2

            cv2.line(
                overlay,
                (0, y),
                (w_img, y),
                (255, 0, 0),
                4
            )

            cv2.putText(
                overlay,
                "SOL",
                (20, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

            
    # display augmented frame
    #cv2.imshow("TensorFlow Detections (Live)", frame)
    #cv2.imshow("EDGES", edges)
    cv2.imshow("Frame + Edges", overlay)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()

