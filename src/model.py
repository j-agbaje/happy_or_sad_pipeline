import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report



def create_tuner():
    """Create hyperparameter tuner"""
    def build_model(hp):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                hp.Int('conv1_filters', 16, 64, step=16), (3,3), 1, 
                activation='relu', input_shape=(256,256,3)
            ),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(
                hp.Int('conv2_filters', 32, 128, step=16), (3,3), 1,
                activation='relu'
            ),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(
                hp.Int('conv3_filters', 64, 256, step=32), 
                (3,3), 1, activation='relu'
            ),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.8, step=0.1)),
            tf.keras.layers.Dense(
                hp.Int('dense_units', 128, 512, step=128), 
                activation='relu'
            ),
            tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.8, step=0.1)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        directory='keras_tuner_results',
        project_name='emotion_cnn'
    )
    
    return tuner


def train_model(train_data, val_data, tuner): 
    tuner.search(train_data, epochs=20, validation_data=val_data)

    model = tuner.get_best_models()[0]
    best_hyperparamters = tuner.get_best_hyerparameters()[0]

    return model

def model_evaluation(model, test_data):
    from sklearn.metrics import classification_report

    y_pred = model.predict(test_data)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    y_true = np.concatenate([y for x,y in test_data], axis=0)

    class_report = classification_report(y_true, y_pred_classes)
    print(class_report)

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='binary')
    recall = recall_score(y_true, y_pred_classes, average='binary')
    f1 = f1_score(y_true, y_pred_classes, average='binary')

    conf_matrix = confusion_matrix(y_true, y_pred_classes)

    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


   






def save_model(model, model_path='models/trained_model.h5'):
    """Save the trained model"""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """Load a trained model"""
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model
