import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from scipy.ndimage import zoom
import keras
from PIL import ImageFont, ImageDraw, Image

# ===== Configuration =====
shape_x = 48
shape_y = 48
labels = ['화남', '혐오', '두려움', '행복', '슬픔', '놀람', '중립']
DISGUST_IDX = labels.index('혐오')  # '혐오' 인덱스
font_path = "C:/Windows/Fonts/malgun.ttf"
font_small = ImageFont.truetype(font_path, 20)
font_big = ImageFont.truetype(font_path, 24)
model = keras.models.load_model('./face_emotion.h5')

# ===== Face Detection Function =====
def detect_face(frame):
    """
    Detects faces in a given video frame using a pre-trained cascade classifier.

    Args:
        frame (np.array): The input video frame.

    Returns:
        tuple: A tuple containing the grayscale frame and a list of detected face coordinates.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(shape_x, shape_y),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Filters detected faces by width to ignore false positives
    valid_faces = [face for face in detected_faces if face[2] > 80]
    return gray, valid_faces

# ===== Face Feature Extraction Function =====
def extract_face_features(gray, detected_faces, offset_coefficients=(0.075, 0.05)):
    """
    Extracts and normalizes face features from the grayscale image.

    Args:
        gray (np.array): The grayscale frame.
        detected_faces (list): A list of detected face coordinates.
        offset_coefficients (tuple): Coefficients for adjusting the face crop area.

    Returns:
        list: A list of normalized face feature arrays.
    """
    face_features_list = []
    for x, y, w, h in detected_faces:
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))
        
        extracted_face = gray[y + vertical_offset:y + h, x + horizontal_offset:x + w - horizontal_offset]
        
        if extracted_face.size == 0:
            continue
        
        new_extracted_face = zoom(
            extracted_face,
            (shape_x / extracted_face.shape[0], shape_y / extracted_face.shape[1])
        )
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) if new_extracted_face.max() != 0 else 1.0
        face_features_list.append(new_extracted_face)
        
    return face_features_list

# ===== Main Loop =====
video_capture = cv2.VideoCapture(0)

# Set webcam resolution to a standard HD size (16:9 ratio)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Desired final video output size while maintaining aspect ratio
base_width = 1300
base_height = 731  # 1300 * 9 / 16

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Force resize the video frame to the desired size while keeping aspect ratio
    resized_frame = cv2.resize(frame, (base_width, base_height))

    gray, detected_faces = detect_face(resized_frame)

    # Convert to PIL Image for drawing Korean text
    frame_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    # Process only if faces are detected
    if len(detected_faces) > 0:
        face_features_list = extract_face_features(gray, detected_faces)
        
        if len(detected_faces) == len(face_features_list):
            for (x, y, w, h), face_zoom in zip(detected_faces, face_features_list):
                face_input = np.reshape(face_zoom, (1, 48, 48, 1))
                pred = model.predict(face_input, verbose=0)  # shape: (1, 7)

                # ===== (핵심) '혐오'를 최종 라벨에서 제외 =====
                masked = pred.copy()
                masked[0, DISGUST_IDX] = -np.inf  # argmax 대상에서 제외
                pred_result = int(np.argmax(masked))

                # ===== (핵심) 표시용 분포에서 '혐오'를 숨김 =====
                shown = pred.copy()
                shown[0, DISGUST_IDX] = 0.0  # 화면 표시에서만 0으로 처리
                total = float(shown.sum())
                if total > 0:
                    shown /= total
                else:
                    # 혹시 모든 값이 0이 되는 드문 수치 문제 대비
                    neutral_idx = labels.index('중립')
                    shown[0, neutral_idx] = 1.0

                # 빨간색 확률 목록 ('혐오'는 스킵)
                yoff = 0
                for i, label in enumerate(labels):
                    if i == DISGUST_IDX:
                        continue
                    draw.text((x + w + 10, y + yoff * 25),
                              f"{label}: {shown[0][i]:.2f}",
                              font=font_small, fill=(255, 0, 0, 0))
                    yoff += 1

                # 초록색 최종 라벨 (혐오가 1등이어도 제외된 상태의 결과)
                # 가독성을 위해 표시 점수는 shown(재정규화된 값) 사용
                draw.text((x, y - 30),
                          f"{labels[pred_result]} {shown[0][pred_result]:.2f}",
                          font=font_big, fill=(0, 255, 0, 0))
    
    # Convert back to OpenCV format
    final_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    # Draw green rectangles around the detected faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(final_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Display the final frame
    cv2.imshow('Video', final_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
