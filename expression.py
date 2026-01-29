import mediapipe as mp
import cv2
import math
from collections import deque
import time
# -------------------- MediaPipe Setup --------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

video = cv2.VideoCapture(0)

# -------------------- Utility --------------------
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# -------------------- Neutral Calibration --------------------
NEUTRAL_FRAMES = 25
neutral_buffer = deque(maxlen=NEUTRAL_FRAMES)
neutral_ready = False
neutral_avg = None

# -------------------- Main Loop --------------------
while True:
    success, frame = video.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    expression = "NEUTRAL"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        # Draw mesh
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(thickness=1),
            mp_drawing.DrawingSpec(thickness=1)
        )

        # -------------------- Normalization --------------------
        face_width = distance(lm[234], lm[454])

        # -------------------- Features --------------------
        mouth_height = distance(lm[13], lm[14]) / face_width
        mouth_width = distance(lm[61], lm[291]) / face_width
        mouth_open_ratio = mouth_height / (mouth_width + 1e-6)

        left_corner_y = lm[61].y
        right_corner_y = lm[291].y
        mouth_center_y = lm[13].y
        mouth_corner_drop = ((left_corner_y + right_corner_y) / 2) - mouth_center_y

        left_eye = distance(lm[159], lm[145]) / face_width
        right_eye = distance(lm[386], lm[374]) / face_width
        eye_open = (left_eye + right_eye) / 2

        left_brow = distance(lm[105], lm[159]) / face_width
        right_brow = distance(lm[334], lm[386]) / face_width
        brow_raise = (left_brow + right_brow) / 2

        features = (mouth_open_ratio, mouth_width, mouth_corner_drop, brow_raise, eye_open)

        # -------------------- Calibration --------------------
        if not neutral_ready:
            neutral_buffer.append(features)
            cv2.putText(frame, "Calibrating NEUTRAL...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if len(neutral_buffer) == NEUTRAL_FRAMES:
                neutral_avg = [
                    sum(f[i] for f in neutral_buffer) / NEUTRAL_FRAMES
                    for i in range(5)
                ]
                neutral_ready = True
            cv2.imshow("Expression Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # -------------------- Delta (Amplified) --------------------
        dm_open, dm_width, dm_corner, dbrow, deye = [
            (features[i] - neutral_avg[i]) * 10 for i in range(5)
        ]

        # -------------------- Expression Logic --------------------

        # SURPRISED
        if dm_open > 0.12 and dbrow > 0.08 and deye > 0.08:
            expression = "SURPRISED"

        # HAPPY
        elif dm_width > 0.08 and dm_corner < -0.04:
            expression = "HAPPY"

        # SAD
        elif dm_corner > 0.06 and dbrow < -0.05:
            expression = "SAD"

        else:
            expression = "NEUTRAL"


    cv2.putText(frame, expression, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imshow("Expression Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f"frame_{int(time.time())}.jpg"
        cv2.imwrite(filename, cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
        print(f"Saved {filename}")
        continue
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
