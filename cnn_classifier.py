import os
import numpy as np
import tensorflow as tf
from PIL import Image

class CNNClassifier:
    def __init__(self, model_path="model_custom.h5", class_file="class_names.txt"):
        # Load nama kelas dari file
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback default
            self.class_names = ["automobile", "cat", "dog", "horse"]

        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Kelas: {self.class_names}")

    def predict(self, pil_image, top_k=3):
        """
        Input  : PIL Image (ukuran bebas)
        Output : list of (label, confidence%) urut dari tertinggi
        """
        # Resize & normalize sesuai input MobileNetV2
        img = pil_image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

        # Prediksi
        preds   = self.model.predict(arr, verbose=0)[0]
        top_idx = preds.argsort()[-top_k:][::-1]

        return [(self.class_names[i], float(preds[i]) * 100)
                for i in top_idx]