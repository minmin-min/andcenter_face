import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from scipy.ndimage import zoom
import keras
from PIL import ImageFont, ImageDraw, Image
import streamlit as st

# ===== 설정 =====
shape_x, shape_y = 48, 48
labels = ['화남', '혐오', '두려움', '행복', '슬픔', '놀람', '중립']
font_path = "C:/Windows/Fonts/malgun.ttf"
font_small = ImageFont.truetype(font_path, 20)
font_big = ImageFont.truetype(font_path, 24)

# ===== 모델 로드 =====
model = keras.models.load_model('./face_emotion.h5')

# ===== 얼굴 검출 =====
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6,
        minSize=(shape_x, shape_y), flags=cv2.CASCADE_SCALE_IMAGE
    )
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            coord.append([x, y, w, h])
    return gray, coord

# ===== 얼굴 특징 추출 =====
def extract_face_features(gray, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in coord:
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

# ===== Streamlit 설정 =====
st.set_page_config(page_title="실시간 감정 인식", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #111; color: white; }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
            margin: 0;
        }
        img { display: block; margin-left: auto; margin-right: auto; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===== 제목 =====
st.markdown("<h1 style='text-align: center; color: lime;'>표정을 지어보세요 😃 </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>인공지능이 실시간으로 감정을 맞춥니다!</h3>", unsafe_allow_html=True)

frame_placeholder = st.empty()

# ===== 카메라 설정 =====
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.error("카메라를 열 수 없습니다.")
        break

    gray, coord = detect_face(frame)

    # PIL 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 감정 분석 및 표시
    face_zoom_list = extract_face_features(gray, coord)
    for (x, y, w, h), face_zoom in zip(coord, face_zoom_list):
        face_input = np.reshape(face_zoom, (1, 48, 48, 1))
        pred = model.predict(face_input, verbose=0)
        pred_result = np.argmax(pred)

        # 감정별 확률 (빨간)
        for i, label in enumerate(labels):
            draw.text((x + w + 10, y + i * 25), f"{label}: {pred[0][i]:.2f}",
                      font=font_small, fill=(255, 0, 0, 0))
        # 최고 감정 (초록)
        draw.text((x, y - 30), f"{labels[pred_result]} {pred[0][pred_result]:.2f}",
                  font=font_big, fill=(0, 255, 0, 0))

        # 초록 박스 OpenCV로 그림
        frame_np = np.array(frame_pil)
        cv2.rectangle(frame_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame_pil = Image.fromarray(frame_np)

    # ===== 패딩 추가 (좌우 + 하단) =====
    frame_np = np.array(frame_pil)
    pad_width_side = 30   # 좌우 패딩
    pad_height_bottom = 0  # 하단 패딩
    pad_height_top = 0    # 상단 패딩
    frame_np = cv2.copyMakeBorder(frame_np, pad_height_top, pad_height_bottom,
                                  pad_width_side, pad_width_side,
                                  cv2.BORDER_CONSTANT, value=(17, 17, 17))

    frame_placeholder.image(frame_np, channels="RGB", use_container_width=True)

video_capture.release()