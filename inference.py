# inference.py
import tensorflow as tf
import numpy as np
import cv2

IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')

# Map class indices to labels (update these based on your training)
class_labels = {0: 'recyclable', 1: 'non_recyclable'}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_labels[predicted_class], confidence

if __name__ == '__main__':
    test_image_path = 'path_to_test_image.jpg'
    label, conf = predict_image(test_image_path)
    print(f"Predicted: {label} (confidence: {conf:.2f})")
