import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from scipy.ndimage import zoom
import keras
from PIL import ImageFont, ImageDraw, Image

shape_x = 48
shape_y = 48

labels = ['화남', '혐오', '두려움', '행복', '슬픔', '놀람', '중립']

font_path = "C:/Windows/Fonts/malgun.ttf"
font_small = ImageFont.truetype(font_path, 20)
font_big = ImageFont.truetype(font_path, 24)

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(shape_x, shape_y),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            coord.append([x, y, w, h])
    return gray, detected_faces, coord

def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:
        x, y, w, h = det
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        if extracted_face.size == 0:
            continue
        new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0], shape_y / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) if new_extracted_face.max() != 0 else 1.0
        new_face.append(new_extracted_face)
    return new_face

model = keras.models.load_model('./face_emotion.h5')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray, detected_faces, coord = detect_face(frame)

    try:
        face_zoom_list = extract_face_features(gray, detected_faces, coord)
        if not face_zoom_list:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1. PIL로 글씨 먼저 그림
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for (x, y, w, h), face_zoom in zip(coord, face_zoom_list):
            face_input = np.reshape(face_zoom, (1, 48, 48, 1))
            pred = model.predict(face_input, verbose=0)
            pred_result = np.argmax(pred)

            # 감정별 확률 빨간 글씨
            for i, label in enumerate(labels):
                draw.text((x + w + 10, y + i*25), f"{label}: {pred[0][i]:.2f}",
                          font=font_small, fill=(255, 0, 0, 0))

            # 최고 확률 감정 초록 글씨
            draw.text((x, y - 30), f"{labels[pred_result]} {pred[0][pred_result]:.2f}",
                      font=font_big, fill=(0, 255, 0, 0))

        # 2. PIL → OpenCV 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 3. OpenCV로 박스 그리기 (글씨 덮어쓰지 않음)
        for (x, y, w, h) in coord:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")
        continue

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
