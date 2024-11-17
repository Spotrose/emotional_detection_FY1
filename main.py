import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path ='model/emotion_model4.h5'
if os.path.exists(model_path):
    print("File size:", os.path.getsize(model_path))
    model = load_model(model_path)  # Load the model here
else:
    print("Error: Model file not found.")

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")

def predict_emotion(frame):
    frame_resized = cv2.resize(frame, (48, 48))
    img_array = img_to_array(frame_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    return emotions[emotion_index]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    emotion = predict_emotion(frame)
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()