# Emotion Detection ML Pipeline

## Video Demo
[YouTube Demo Link](https://youtu.be/RfDLKSG7FnM)


## Project Description

A machine learning pipeline for emotion detection from facial images. The system classifies images as "happy" or "sad" using a Convolutional Neural Network (CNN) with capabilities for real-time prediction, bulk data upload, and model retraining.

### Key Features
- **Image Classification**: Binary emotion detection (happy/sad)
- **Real-time Prediction**: Upload single images for instant classification
- **Bulk Data Upload**: Upload multiple images with class labels for retraining
- **Model Retraining**: Retrain the model with new data using existing model as base
- **RESTful API**: FastAPI backend with automatic documentation
- **Database Storage**: MongoDB for data storage

### Technology Stack
- **Backend**: FastAPI, TensorFlow/Keras
- **Database**: MongoDB with GridFS for image storage
- **ML Framework**: TensorFlow 2.x, Keras Tuner
- **Image Processing**: PIL, OpenCV, Tensorflow


## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/emotion-detection-pipeline.git
cd emotion-detection-pipeline
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
MONGO_URI=your_mongodb_connection_string
```

### 4. Directory Structure
Ensure your project follows this structure:
```
emotion-detection-pipeline/
│
├── README.md
├── requirements.txt
├── app.py                 # FastAPI application
├── .env
│
├── notebook/
│   └── emotion_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── data/
│   ├── train/
│   └── test/
│
└── models/
    └── emotion_model.keras
```

### 5. Run the Application
```bash
python app.py
```

### 6. Access Swagger UI
Navigate to `http://localhost:8000/docs` to access the interactive API documentation.

## API Endpoints

### Testing the Functions

1. **Welcome Endpoint**
   - **GET** `/`
   - Test basic connectivity

2. **Single Image Prediction**
   - **POST** `/predict`
   - Upload an image file to get emotion prediction
   - Returns: predicted class and confidence score

3. **Bulk Image Upload**
   - **POST** `/upload_bulk`
   - Upload multiple images with class labels (0=sad, 1=happy)
   - Images stored in MongoDB for future retraining

4. **Model Retraining**
   - **POST** `/retrain`
   - Triggers retraining using uploaded images from MongoDB
   - Uses existing model as starting point with lower learning rate

### Model Architecture
- Convolutional Neural Network (CNN)
- Input: 256x256x3 RGB images
- 3 Convolutional layers with MaxPooling
- Dropout layers for regularization
- Binary classification output (sigmoid activation)
