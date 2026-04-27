import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#  FIXED PATH (IMPORTANT)
model_path = r"C:\Users\sanjy\OneDrive\Documents\project\hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect hand
    result = detector.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape
            x = int(hand_landmarks[8].x * w)
            y = int(hand_landmarks[8].y * h)

            index_up = hand_landmarks[8].y < hand_landmarks[6].y
            middle_up = hand_landmarks[12].y < hand_landmarks[10].y

            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            #  DRAW MODE
            if index_up and not middle_up:
                if prev_x == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                prev_x, prev_y = x, y

            #  ERASE MODE
            elif index_up and middle_up:
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    # combine canvas + frame
    combined = cv2.add(frame, canvas)

    cv2.imshow("Virtual Drawing Board", combined)

    key = cv2.waitKey(1)

    # clear screen
    if key == ord('c'):
        canvas = np.zeros_like(frame)

    # exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()