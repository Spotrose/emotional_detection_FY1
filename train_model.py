# Environment
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import kagglehub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Download latest version of dataset
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

train_dir = '/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train'
test_dir = '/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test'

# Display dataset stats
print("Training samples:", len(os.listdir(train_dir)))
print("Testing samples:", len(os.listdir(test_dir)))

# Preview an example image
img = cv2.imread(f'{train_dir}/angry/Training_10118481.jpg')
plt.imshow(img)
plt.show()

# Image Data Generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb'
)

# Model Building
base_model = ResNet50(input_shape=(48, 48, 3), include_top=False, weights='imagenet')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(base_model)
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(48, 48, 4)))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=3, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Model Compilation
model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callback to stop training if accuracy reaches 99.9%
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy, stopping training!")
            self.model.stop_training = True

# Model Training
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    shuffle=True,
    callbacks=[MyCallback()]
)

# Evaluation on validation data
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Save the model
model.save('models/emotion_model.h5')

# Plotting Training History
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
