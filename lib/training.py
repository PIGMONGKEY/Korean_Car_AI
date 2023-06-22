import os

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/test"
validation_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/validation"

image_size = (512, 384)
batch_size = 64
num_classes = 100
epochs = 2

MODEL_SAVE_FOLDER_PATH = './model/'


def create_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 384, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

data_generator = ImageDataGenerator(rescale=1. / 255)

train_data = data_generator.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_data = data_generator.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = create_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_path = MODEL_SAVE_FOLDER_PATH + 'KoreanCar-' + '{epoch:02d}-{val_loss:4f}.hdf5'
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                                   verbose=1, save_best_only=True)

history = model.fit(
    train_data,
    epochs=epochs,
    verbose=1,
    callbacks=cb_checkpoint,
    validation_data=validation_data
)

model.save(MODEL_SAVE_FOLDER_PATH + 'korean_car_model.hdf5')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
