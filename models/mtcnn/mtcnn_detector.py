# models/mtcnn/mtcnn_detector.py

from facenet_pytorch import MTCNN
import torch
from PIL import Image


class FaceDetector:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)

    def detect_and_align(self, image_path):
        """
        Detects and aligns the most prominent face in an image.
        :param image_path: Path to the input image.
        :return: PIL image of the aligned face, or None if no face is found.
        """
        image = Image.open(image_path).convert('RGB')
        face = self.mtcnn(image)

        if face is None:
            print("❌ No face detected.")
            return None

        print("✅ Face detected and aligned.")
        return face  # Tensor shape: (3, 160, 160)
