import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path = 'model/emotion_model4.h5'
if os.path.exists(model_path):
    print("File size:", os.path.getsize(model_path))
    model = load_model(model_path)  # Load the model here
else:
    messagebox.showerror("Error", "Model file not found.")
    exit()

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define emotion detection function
def start_emotion_detection():
    cap = cv2.VideoCapture(0)  # Open video capture
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

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
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the GUI window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("400x200")

# Add a label
label = tk.Label(root, text="Click 'Start' to begin emotion detection", font=("Arial", 14))
label.pack(pady=30)

# Add a Start button to launch the emotion detection
start_button = tk.Button(root, text="Start", command=start_emotion_detection, bg="green", fg="white", font=("Arial", 12))
start_button.pack(pady=10)

label = tk.Label(root, text="Press 'Q' to End emotion detection", font=("Arial", 14))
label.pack(pady=8)

# Run the Tkinter event loop
root.mainloop()
