import cv2
import numpy as np
from face_recognition_arc import get_live_embedding, verify_face
from gesture_liveness import run_gesture_challenge

stored_embedding = np.load("embeddings/user_embedding.npy")

cap = cv2.VideoCapture(0)
kyc_verified = False

print("Starting KYC verification...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    live_embedding = get_live_embedding(frame)

    if live_embedding is not None:
        if verify_face(live_embedding, stored_embedding):
            print("Face verified. Starting liveness check...")
            if run_gesture_challenge():
                kyc_verified = True
                break

    cv2.imshow("KYC Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if kyc_verified:
    print("KYC VERIFIED")
else:
    print("KYC FAILED")
