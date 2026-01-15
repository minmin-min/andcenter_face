# list_cams.py
import cv2

def available_cams(max_index=10):
    found = []
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows 권장
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)  # fallback
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
            cap.release()
    return found

print("Available camera indices:", available_cams(5))
