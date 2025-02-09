#detect_barcode.py
import cv2
import numpy as np
import tensorflow as tf
from pyzbar.pyzbar import decode

# Load the trained model
model = tf.keras.models.load_model('barcode_detector.h5')

# Open camera connection
camera = cv2.VideoCapture(0)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return thresholded

def rotate_image(image):
    rotations = [image]
    for _ in range(3):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotations.append(image)
    return rotations

while True:
    ret, frame = camera.read()
    if not ret:
        break

    preprocessed_frame = preprocess_image(frame)
    rotated_frames = rotate_image(preprocessed_frame)

    barcode_detected = False

    for rotated_frame in rotated_frames:
        barcodes = decode(rotated_frame)
        if barcodes:
            barcode_detected = True
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                print(f"Detected {barcode_type} barcode: {barcode_data}")

                rect_points = barcode.polygon
                if len(rect_points) == 4:
                    pts = np.array(rect_points, dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                else:
                    (x, y, w, h) = barcode.rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    if not barcode_detected:
        print("No barcode detected")

    cv2.imshow("Barcode Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
