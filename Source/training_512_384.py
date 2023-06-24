import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
os.environ['TF_MACOS_GPU_LIBRARY_PATH'] = '/System/Library/Frameworks/Metal.framework/Versions/Current/Frameworks/\
    GPUCompiler.framework/Versions/Current/lib/libgpucompiler.dylib'

print(tf.config.list_physical_devices())

train_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/train"
validation_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/validation"

image_size = (384, 512)
batch_size = 32
num_classes = 100
epochs = 10

MODEL_SAVE_FOLDER_PATH = '../lib/model/'


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(512, 384, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
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
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True)

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
