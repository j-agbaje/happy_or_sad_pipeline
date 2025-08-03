from fastapi import FastAPI, UploadFile, HTTPException, File, APIRouter
from src.preprocessing import *
from src.model import *
from src.prediction import *
import os
from contextlib import asynccontextmanager
import tensorflow as tf
from pymongo.mongo_client import MongoClient
import gridfs 
from bson import ObjectId
from typing import List 
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import io
load_dotenv()


uri = os.getenv("MONGO_URI")





# Create a new client and connect to the server
client = MongoClient(uri)

db = client.happy_sad_db
fs = gridfs.GridFS(db)


# collection = db['emotion_data']



print("Model exists?", os.path.exists('./models/emotion_model.keras'))



# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load model on startup
#     global model
#     try:
#         model = load_model('./models/emotion_model.keras')
#         print("Model load successful")
#     except Exception as e:
#         print(f'Model not loaded: {e}')
#     yield
    # Cleanup on shutdown (if needed)


app = FastAPI()

@app.get('/')
def welcome():
    return {"welcome message": "welcome"}


# @app.post('/upload_images')
# async def upload_image(file: UploadFile = File(...)):
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="Expected an image file")
#     contents = await file.read()
#     file_id = fs.put(contents, filename=file.filename)

#     return {"message": "Image uploaded successfully", "file_id": str(file_id)}


@app.post('/upload_bulk')
async def upload_bulk_to_mongo(files: List[UploadFile] = File(...), class_label: int = 0 ):
    uploaded_files = 0


    for file in files:
        if file.content_type.startswith('image/'):
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"

            file.file.seek(0)

            fs.put(
                file.file,
                filename=filename,
                content_type = file.content_type,
                class_label = class_label
            )

            uploaded_files += 1
    return {
    "message": f"{uploaded_files} images uploaded to MongoDB",
    "class_label": class_label
}





@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    model1 = tf.keras.models.load_model('./models/emotion_model.keras')
    print(f"Model in endpoint: {model1}")

    if model1 == None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    
    
    filename = file.filename
    

    image_data = await file.read()
    os.makedirs("uploaded_images", exist_ok=True)
    image_path = os.path.join('uploaded_images', filename)
    

    with open(f'uploaded_images/{filename}', 'wb') as f:
        f.write(image_data)

    probs, result = predict_single_image(image_path, model1)

    os.remove(image_path)

    return {
        "filename" : file.filename,
        "predicted_class": result['predicted_class'],
        "confidence": result['confidence']
    }


def load_images_from_mongo():
    images = []
    labels = []

    for grid_file in fs.find():
        image_data = grid_file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((256, 256))

        if image.mode != "RGB":
            image = image.convert('RGB')

        img_array = np.array(image) / 255.0
        images.append(img_array)
        labels.append(grid_file.class_label)
    
    return np.array(images), np.array(labels)
    

@app.post("/retrain")
async def retrain_model():
    """Retrain model with MongoDB data"""
    if not os.path.exists('./models/emotion_model.keras'):
        raise HTTPException(status_code=404, detail="Base model not found")
    
    # Load data from MongoDB
    images, labels = load_images_from_mongo()
    
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images found in MongoDB")
    
    # Load existing model
    model = tf.keras.models.load_model('./models/emotion_model.keras')
    
    # Split data
    split_idx = int(0.8 * len(images))
    indices = np.random.permutation(len(images))
    
    train_images = images[indices[:split_idx]]
    train_labels = labels[indices[:split_idx]]
    val_images = images[indices[split_idx:]]
    val_labels = labels[indices[split_idx:]]
    
    # Retrain with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=10,
        batch_size=32
    )
    
    # Save retrained model
    model.save('./models/retrained_emotion_model.keras')
    
    return {
        "message": "Model retrained successfully",
        "training_samples": len(train_images),
        "validation_samples": len(val_images),
        "final_accuracy": float(history.history['val_accuracy'][-1])
    }





