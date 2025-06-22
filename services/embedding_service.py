# services/embedding_service.py

import numpy as np
from keras_facenet import FaceNet
import torch

class EmbeddingService:
    def __init__(self):
        """
        Initializes the FaceNet embedder from keras-facenet.
        """
        self.embedder = FaceNet()
        print("✅ keras-facenet embedder loaded.")

    def preprocess(self, face_tensor):
        """
        Converts and normalizes a face tensor for keras-facenet.
        :param face_tensor: Torch tensor of shape (3, 160, 160)
        :return: Numpy array of shape (160, 160, 3)
        """
        if isinstance(face_tensor, torch.Tensor):
            face = face_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
        else:
            raise TypeError("face_tensor must be a torch.Tensor")

        # Convert to uint8 if needed (FaceNet can handle raw RGB)
        face = np.clip(face, 0, 255).astype(np.uint8)
        return face

    def get_embedding(self, face_tensor):
        """
        Generates the 512D embedding using keras-facenet.
        :param face_tensor: Aligned face tensor (3, 160, 160)
        :return: 1D list of 512 floats
        """
        face_img = self.preprocess(face_tensor)
        embedding = self.embedder.embeddings([face_img])[0]
        return embedding.tolist()
