# 👤 Face Recognition System with Live Detection

A full-fledged face recognition system using **MTCNN** for face detection and **FaceNet** (via `keras-facenet`) for face embeddings. This system supports both face registration and recognition using a webcam — all in real time with no manual intervention like pressing a key.

---

## 📊 Pipeline Overview

```text
[Live Webcam Feed]
       ⬇
[MTCNN Face Detection & Alignment]
       ⬇
[FaceNet Embedding Extraction]
       ⬇
[Registration] ➡ Saves embedding & aligned image in JSON DB
       ⬇
[Recognition] ➡ Compares live embedding with DB using cosine similarity
```


# run  
``` pip install -r requirements.txt ```


# Run the Application
``` python app.py ```

# 🔍 Features
✅ Live face detection (no SPACE key required)
✅ JSON-based face database
✅ Cosine similarity matching
✅ Clear and modular structure
✅ Easy to extend with GUI/API support


# ✅ Requirements
Python 3.7+

OpenCV

NumPy

Keras + TensorFlow

scikit-learn

Pillow

keras-facenet

MTCNN (local/custom version)



