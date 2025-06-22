import os
import json
import time
import cv2
from pathlib import Path
from datetime import datetime
from torchvision.transforms import ToPILImage

from models.mtcnn.mtcnn_detector import FaceDetector
from services.embedding_service import EmbeddingService

DB_PATH = "db/face_db.json"
ALIGNED_FACE_DIR = "data/aligned_faces"
RAW_FACE_DIR = "data/raw_faces"


class RegistrationService:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = EmbeddingService()

        Path(ALIGNED_FACE_DIR).mkdir(parents=True, exist_ok=True)
        Path(RAW_FACE_DIR).mkdir(parents=True, exist_ok=True)
        Path("db").mkdir(parents=True, exist_ok=True)

    def load_database(self):
        if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
            with open(DB_PATH, 'r') as file:
                return json.load(file)
        else:
            return {}

    def save_database(self, database):
        with open(DB_PATH, 'w') as file:
            json.dump(database, file, indent=4)

    def capture_face_automatically(self, save_path):
        cap = cv2.VideoCapture(0)
        print("📸 Opening camera... Looking for a face...")

        start_time = time.time()
        detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Show frame
            cv2.imshow("Auto Face Capture - Look into the camera", frame)
            key = cv2.waitKey(1) & 0xFF

            # Try to detect and align face
            temp_image_path = "temp_raw.jpg"
            cv2.imwrite(temp_image_path, frame)
            aligned_face = self.detector.detect_and_align(temp_image_path)

            if aligned_face is not None:
                cv2.imwrite(save_path, frame)
                print("✅ Face detected and captured.")
                detected = True
                break

            if time.time() - start_time > 10:
                print("⏰ Timeout: No face detected in 10 seconds.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return detected

    def register_user(self, user_name):
        print(f"🔍 Registering user: {user_name}")

        # Step 1: Capture image automatically
        raw_image_path = os.path.join(RAW_FACE_DIR, f"{user_name}.jpg")
        success = self.capture_face_automatically(raw_image_path)
        if not success:
            print("❌ Registration failed: No face captured.")
            return False

        # Step 2: Detect and align face
        aligned_face_tensor = self.detector.detect_and_align(raw_image_path)
        if aligned_face_tensor is None:
            print("❌ Registration failed: No face detected.")
            return False

        # Step 3: Save aligned face
        aligned_image_path = os.path.join(ALIGNED_FACE_DIR, f"{user_name}.jpg")
        ToPILImage()(aligned_face_tensor).save(aligned_image_path)

        # Step 4: Generate embedding
        embedding = self.embedder.get_embedding(aligned_face_tensor)

        # Step 5: Save to DB
        database = self.load_database()
        database[user_name] = {
            "embedding": embedding,
            "image": aligned_image_path,
            "registered_on": datetime.now().isoformat()
        }
        self.save_database(database)

        print(f"✅ {user_name} registered successfully.")
        return True


# 🎯 CLI entry point
if __name__ == "__main__":
    reg = RegistrationService()
    name = input("👤 Enter user name: ").strip()
    reg.register_user(name)
