import cv2
import numpy as np

# יצירת אובייקט המצלמה
cap = cv2.VideoCapture(0)

# מודל DNN לזיהוי פנים
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# משתנה לשמירת מזהי פנים שזוהו כבר
detected_faces = set()
saved_histograms = []  # רשימה לשמירת ההיסטוגרמות של תמונות שנשמרו
face_id = 0  # משתנה לספירה ולמתן מזהה לכל פנים שנשמר

def calculate_histogram(image):
    """פונקציה לחישוב היסטוגרמת הצבעים של תמונה"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def is_similar_to_saved_histograms(new_hist, saved_histograms, threshold=0.6):
    """בודק אם תמונה חדשה דומה לאחת מהתמונות שנשמרו לפי היסטוגרמה"""
    for saved_hist in saved_histograms:
        similarity = cv2.compareHist(new_hist, saved_hist, cv2.HISTCMP_CORREL)
        if similarity >= threshold:
            return True
    return False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # זיהוי פנים באמצעות הרשת הנוירונית
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # סינון לפי רמת ביטחון מסוימת (למשל 0.6)
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            face_position = (x // 10, y // 10, (x2 - x) // 10, (y2 - y) // 10)

            if face_position not in detected_faces:
                detected_faces.add(face_position)
                face_image = frame[y:y2, x:x2]

                # חישוב היסטוגרמת התמונה החדשה
                new_hist = calculate_histogram(face_image)

                # בדיקה אם התמונה דומה לתמונות שכבר נשמרו
                if not is_similar_to_saved_histograms(new_hist, saved_histograms):
                    face_id += 1
                    saved_histograms.append(new_hist)
                    cv2.imwrite(f'face_{face_id}.png', face_image)
                    print(f"Saved face {face_id} to face_{face_id}.png")
                else:
                    print("Face already exists, not saving.")

            # ציור ריבוע סביב הפנים שזוהו
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    # הצגת התמונה עם הפנים המזוהות
    cv2.imshow('Face Detection', frame)

    # יציאה בלחיצה על מקש ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# שחרור המצלמה וסגירת החלונות
cap.release()
cv2.destroyAllWindows()
