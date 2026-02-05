import cv2
import json
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view (optional but recommended)
    frame = cv2.flip(frame, 1)

    # --- ROI (Region of Interest) ---
    h, w, _ = frame.shape
    size = 300
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Preprocessing (MATCHES TRAINING) ---
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)

    # --- Prediction ---
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    label = f"{class_names[class_id]}  ({confidence*100:.1f}%)"

    # Display prediction
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
