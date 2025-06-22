# services/recognition_service.py

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from models.mtcnn.mtcnn_detector import FaceDetector
from services.embedding_service import EmbeddingService

DB_PATH = "db/face_db.json"
SIMILARITY_THRESHOLD = 0.7
TEMP_TEST_IMG = "test_images/temp_recognition.jpg"

class RecognitionService:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = EmbeddingService()
        self.database = self.load_database()
        Path("test_images").mkdir(parents=True, exist_ok=True)

    def load_database(self):
        try:
            with open(DB_PATH, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print("⚠️ Face DB not found.")
            return {}

    def capture_face_automatically(self, save_path=TEMP_TEST_IMG):
        cap = cv2.VideoCapture(0)
        print("📸 Opening camera... Looking for a face...")

        start_time = time.time()
        detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera.")
                break

            cv2.imshow("Auto Face Recognition - Look into the camera", frame)
            key = cv2.waitKey(1) & 0xFF
            temp_image_path = "temp_recog_live.jpg"
            cv2.imwrite(temp_image_path, frame)

            # Try detecting face
            aligned_face = self.detector.detect_and_align(temp_image_path)
            if aligned_face is not None:
                cv2.imwrite(save_path, frame)
                print("✅ Face detected and captured for recognition.")
                detected = True
                break

            if key == 27:  # ESC to cancel
                print("❌ Recognition cancelled.")
                break

            if time.time() - start_time > 10:
                print("⏰ Timeout: No face detected in 10 seconds.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return detected

    def recognize_user(self, image_path):
        print(f"🔍 Recognizing face from {image_path}")

        aligned_face = self.detector.detect_and_align(image_path)
        if aligned_face is None:
            return {"identity": None, "status": "no_face_detected"}

        test_embedding = np.array(self.embedder.get_embedding(aligned_face)).reshape(1, -1)

        best_match = None
        highest_score = 0.0

        for name, data in self.database.items():
            db_embedding = np.array(data["embedding"]).reshape(1, -1)
            score = cosine_similarity(test_embedding, db_embedding)[0][0]

            if score > highest_score:
                highest_score = score
                best_match = name

        if best_match and highest_score >= SIMILARITY_THRESHOLD:
            return {
                "identity": best_match,
                "similarity": round(float(highest_score), 4),
                "status": "match_found"
            }
        else:
            return {
                "identity": "Unknown",
                "similarity": round(float(highest_score), 4),
                "status": "no_match"
            }


# 🎯 Entry Point
if __name__ == "__main__":
    recog = RecognitionService()

    if recog.capture_face_automatically():
        result = recog.recognize_user(TEMP_TEST_IMG)

        print("\n🧪 Recognition Result:")
        print(f"🔹 Status     : {result['status']}")
        print(f"🔹 Identity   : {result['identity']}")
        print(f"🔹 Similarity : {result.get('similarity', 'N/A')}")

        if result["status"] == "match_found":
            print("\n✅ Face confirmed. You are Authenticated.")
        elif result["status"] == "no_face_detected":
            print("\n❌ No face detected.")
        else:
            print("\n❌ Face not found in DB. Not Allowed.")
    else:
        print("\n⚠️ Recognition aborted.")
