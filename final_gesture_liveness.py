import cv2
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ---------------- MODEL ----------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- GESTURES ----------------
GESTURES = {
    "FIST": 0,
    "ONE_FINGER": 1,
    "TWO_FINGERS": 2,
    "THREE_FINGERS": 3,
    "FOUR_FINGERS": 4,
    "OPEN_PALM": 5
}

current_challenge = random.choice(list(GESTURES.keys()))
verified = False

print("Gesture challenge:", current_challenge)

# ---------------- STABILITY BUFFER ----------------
finger_history = deque(maxlen=10)

# ---------------- STRONG FINGER COUNT ----------------
def count_fingers(hand_landmarks, handedness):
    fingers = 0

    # ---- Thumb (X axis) ----
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]

    if handedness == "Right":
        if thumb_tip.x > thumb_ip.x:
            fingers += 1
    else:  # Left hand
        if thumb_tip.x < thumb_ip.x:
            fingers += 1

    # ---- Other fingers (Y axis) ----
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers += 1

    return fingers

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks and result.handedness:
        hand_landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name

        finger_count = count_fingers(hand_landmarks, handedness)
        finger_history.append(finger_count)

        # ---- Stability Check ----
        stable_count = max(set(finger_history), key=finger_history.count)

        if stable_count == GESTURES[current_challenge]:
            verified = True

    # ---------------- UI ----------------
    if verified:
        text = f"{current_challenge} VERIFIED"
        color = (0, 255, 0)
    else:
        text = f"Show: {current_challenge}"
        color = (0, 0, 255)

    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("STRONG Gesture Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or verified:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
