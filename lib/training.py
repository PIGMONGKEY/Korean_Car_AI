import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/train"
validation_data_dir = "/Users/pigmong0202/KoreanCar_DataSets/512_384/validation"

image_size = (512, 384)
batch_size = 32
num_classes = 100
epochs = 10

data_generator = ImageDataGenerator(rescale=1./255)

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

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 384, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=validation_data
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
