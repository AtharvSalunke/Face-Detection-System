# app.py

from services.registration_service import RegistrationService
from services.recognition_service import RecognitionService

def main():
    while True:
        print("\n==============================")
        print("🎯 Face Recognition System")
        print("==============================")
        print("1️⃣ Register New Face")
        print("2️⃣ Recognize Face")
        print("3️⃣ Exit")
        choice = input("👉 Enter your choice (1/2/3): ").strip()

        if choice == '1':
            reg_service = RegistrationService()
            name = input("👤 Enter user name: ").strip()
            if not name:
                print("❌ Username cannot be empty.")
                continue
            success = reg_service.register_user(name)
            if success:
                print(f"✅ Face for '{name}' registered successfully.")
            else:
                print("❌ Registration failed. No face was detected.")

        elif choice == '2':
            recog_service = RecognitionService()

            if not recog_service.database:
                print("\n⚠️ No registered users found in the database.")
                print("👉 Please register a face first before attempting recognition.")
                continue

            temp_image_path = "test_images/temp_recognition.jpg"
            success = recog_service.capture_face_automatically(temp_image_path)
            if not success:
                print("❌ No face detected for recognition.")
                continue

            result = recog_service.recognize_user(temp_image_path)

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

        elif choice == '3':
            print("👋 Exiting. Have a great day!")
            break

        else:
            print("❌ Invalid input. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
