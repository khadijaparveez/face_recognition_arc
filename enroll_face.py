import cv2
import numpy as np
import insightface
import os

os.makedirs("embeddings", exist_ok=True)

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)

print("Press 'c' to capture face")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Enroll Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        if faces:
            embedding = faces[0].embedding
            np.save("embeddings/user_embedding.npy", embedding)
            print("Face enrolled successfully")
            break

cap.release()
cv2.destroyAllWindows()
