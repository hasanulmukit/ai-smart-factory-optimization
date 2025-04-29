# data_preprocessing.py
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths (adjust these paths based on your dataset location)
DATA_DIR = './dataset'
IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_path):
    # Load image in color
    img = cv2.imread(image_path)
    # Resize image
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize image [0, 1]
    img = img.astype("float32") / 255.0
    return img

def create_data_generators(batch_size=32):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2  # 20% for validation
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

if __name__ == '__main__':
    train_gen, val_gen = create_data_generators()
    # Visualize a batch of images
    images, labels = next(train_gen)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()
