import os
import cv2
import imghdr
import tensorflow as tf
import hashlib
import numpy as np

def remove_unauthorized_extensions(path_to_images):
    data_dir = path_to_images
    image_exts = ['jpeg', 'jpg', 'png']

    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir,image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)

            except Exception as e:
                print('Issue with image {}'.format(image_path))





def batch_load_data(images_path):
    remove_unauthorized_extensions(images_path)
    data = tf.keras.utils.image_dataset_from_directory(images_path)
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    
    return data, batch


def normalize_image(image, label):
     image = tf.cast(image, tf.float32) / 255.0
     return image, label


def random_horizontal_flip(image):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

def random_vertical_flip(image):
     # Random vertical flip
    image = tf.image.random_flip_up_down(image)

def random_brightness():
    # Random brightness
    image = tf.image.random_brightness(image, 0.3)

def random_rotation():
    # Random rotation (-30 to +30 degrees)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))





def hash_image(path):
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    



def find_and_remove_duplicates(train_dir, test_dir):
    train_hashes = {}

    # Step 1: Hash all training images
    for fname in os.listdir(train_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(train_dir, fname)
            train_hashes[hash_image(path)] = path

    # Step 2: Compare with test images
    removed = 0
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(test_dir, fname)
            h = hash_image(path)
            if h in train_hashes:
                os.remove(path)
                removed += 1
                print(f"🗑️ Removed duplicate from test: {path}")

    print(f"✅ Removed {removed} duplicate(s) from: {test_dir}")




def augment_images(image, label):
    
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, 0.3)
    
    # Random rotation (-30 to +30 degrees)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    # Normalize
    image = tf.cast(image, tf.float32) / 255
    
    return image, label


def delete_ds_store(start_path):
    deleted = 0
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file == '.DS_Store':
                path = os.path.join(root, file)
                os.remove(path)
                deleted += 1
                print(f"🗑️ Deleted: {path}")
    print(f"\n✅ Deleted {deleted} .DS_Store file(s).")


def preprocess_single_image(image_path, img_size=(256,256), interpolation=cv2.INTER_LINEAR):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError('Could not read image path')
    img = cv2.resize (img, img_size, interpolation=interpolation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')/255.0
    return np.expand_dims(img, axis=0)

def create_data_pipeline():
    """loads data from files"""
    pass

def display_data(data):
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    return batch



