import tensorflow as tf
from src.preprocessing import preprocess_single_image
import numpy as np
import os

def load_model(model_path):
    tf.keras.models.load_model(model_path)

def predict_single_image(image_path, model):
    if model is None:
        raise ValueError('Model not loaded please load model first or provide model as an input')
    
    processed = preprocess_single_image(image_path)

    probs = model.predict(processed)

    if probs.ndim != 2 or probs.shape[1] not in (1, 2):
        raise ValueError(f"Unexpected output shape {probs.shape}; expected (batch,1) or (batch,2).")

    p = float(probs[0][0] if probs.shape[1] == 1 else max(probs[0]))

    idx = int(p >= 0.5) if probs.shape[1] == 1 else int(np.argmax(probs[0]))
    confidence = p if idx == 1 else (1.0 - p if probs.shape[1] == 1 else float(probs[0][idx]))
    label = "Happy" if idx == 1 else "Sad"
    results = {"predicted_class": label, "confidence": confidence}
    return probs, results
    

def predict_multiple_images(model, image_folder):
    exts = ('.jpg','.jpeg','.png','.bmp')
    if model is None:
        raise ValueError('Model not loaded please load model first or provide model as an input')
    
    if not os.path.isdir(image_folder):
        raise ValueError(f"Expected a directory, got: {image_folder}")
    
    results = []
    for fname in sorted(os.listdir(image_folder)):
            if not fname.lower().endswith(exts):
                continue  # skip non-image files

            path = os.path.join(image_folder, fname)
            try:
                # Correctly pass both required args
                result = predict_single_image(path, model)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                results.append({'image_path': path, 'error': str(e)})

        
    return results


def predict_from_arrays():
    pass

