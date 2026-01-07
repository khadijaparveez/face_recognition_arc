# face_recognition_arc
🔐 AI-Based KYC Face Recognition & Liveness Verification System
📌 Project Name

kyc-liveness-verification

A real-time KYC (Know Your Customer) verification system that uses ArcFace-based face recognition combined with gesture-based liveness detection to ensure that a real, live user is present during verification.

📖 Overview

This project implements a prototype-level KYC liveness verification pipeline using a webcam.
It verifies both:

Identity – by matching the user’s face using ArcFace embeddings

Liveness – by asking the user to perform a random hand gesture

This approach helps prevent photo spoofing and basic replay attacks, which are common risks in online identity verification systems.

✨ Features

👤 Real-time face detection and recognition

🧬 ArcFace (InsightFace) based face embeddings (512-D)

✋ Gesture-based liveness verification using MediaPipe Hands

🎲 Random gesture challenge per verification attempt

⏱ Timeout-controlled liveness check

🔐 Final KYC PASS / FAIL decision

🧩 Modular and extensible Python codebase

🧠 System Architecture

Face Enrollment

Capture user’s face

Extract ArcFace embedding

Store embedding for future verification

Face Verification

Capture live face

Generate ArcFace embedding

Compare with stored embedding using cosine similarity

Liveness Verification

Random gesture challenge is generated

User performs gesture within time limit

Gesture is validated using hand landmark detection

Final Decision

KYC is marked VERIFIED only if:

Face matches AND

Liveness challenge succeeds

🛠 Technology Stack

Python 3

OpenCV

NumPy

InsightFace (ArcFace)

ONNX Runtime

MediaPipe (Hand Landmarker)

📂 Project Structure
kyc-liveness-verification/
│
├── enroll_face.py              # One-time user face enrollment
├── face_recognition_arc.py     # ArcFace recognition logic
├── gesture_liveness.py         # Gesture-based liveness detection
├── final_kyc.py                # Main KYC verification flow
├── requirements.txt
└── embeddings/
    └── user_embedding.npy      # Stored facial embedding

⚙️ Installation
1️⃣ Install Dependencies
pip install -r requirements.txt

requirements.txt
opencv-python
numpy
insightface
onnxruntime
mediapipe

▶️ How to Run
Step 1️⃣ Face Enrollment (Run Once)

Registers the user’s face and stores the ArcFace embedding.

python enroll_face.py


Press c to capture the face

Embedding is saved to embeddings/user_embedding.npy

Step 2️⃣ Run KYC Verification
python final_kyc.py


Flow:

Live face is captured

Face is verified using ArcFace

Gesture liveness challenge starts

Final KYC result is displayed in terminal

📌 Gesture Liveness Details

Gestures supported:

OPEN_PALM

FIST

ONE_FINGER

TWO_FINGERS

THREE_FINGERS

One gesture is randomly selected per session

User must complete gesture within the timeout period

Hand landmarks are detected using MediaPipe

Download hand_landmarker.task from MediaPipe and place it in the project root.



🏁 Conclusion

This project demonstrates a modular, real-time KYC liveness verification system using ArcFace face recognition and gesture-based liveness detection.
It provides a strong foundation that can be extended into enterprise-grade onboarding and identity verification solutions.
