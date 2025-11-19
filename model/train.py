import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

DATASET_DIR = "data"

def load_data():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1/255.)

    train = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset="training"
    )

    val = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset="validation"
    )

    return train, val

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model():
    train, val = load_data()
    model = build_model()

    checkpoint = ModelCheckpoint("model.h5", save_best_only=True, monitor="val_accuracy")

    history = model.fit(
        train,
        validation_data=val,
        epochs=10,
        callbacks=[checkpoint]
    )

    print("Training complete. Model saved as model.h5")

if __name__ == "__main__":
    train_model()
