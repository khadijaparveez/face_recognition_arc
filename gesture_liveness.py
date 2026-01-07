def run_gesture_challenge(timeout_seconds=8):

    import cv2
    import random
    import time
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

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
    cap = cv2.VideoCapture(0)

    GESTURES = ["OPEN_PALM", "FIST", "ONE_FINGER", "TWO_FINGERS", "THREE_FINGERS"]
    current_challenge = random.choice(GESTURES)
    verified = False
    start_time = time.time()

    def count_extended_fingers(hand_landmarks):
        tips = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]
        count = 0
        for tip, base in zip(tips, bases):
            if hand_landmarks[tip].y < hand_landmarks[base].y:
                count += 1
        return count

    print("Show gesture:", current_challenge)

    while True:
        if time.time() - start_time > timeout_seconds:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            fingers = count_extended_fingers(hand_landmarks)

            if current_challenge == "OPEN_PALM" and fingers >= 4:
                verified = True
            elif current_challenge == "FIST" and fingers == 0:
                verified = True
            elif current_challenge == "ONE_FINGER" and fingers == 1:
                verified = True
            elif current_challenge == "TWO_FINGERS" and fingers == 2:
                verified = True
            elif current_challenge == "THREE_FINGERS" and fingers == 3:
                verified = True

        text = f"{current_challenge} VERIFIED" if verified else f"Show: {current_challenge}"
        color = (0, 255, 0) if verified else (0, 0, 255)

        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Gesture Liveness", frame)

        if verified or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return verified
