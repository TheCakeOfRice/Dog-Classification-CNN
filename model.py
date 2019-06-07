import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

# Data extraction, splitting into training and validation sets, and resizing.
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)
train_gen = datagen.flow_from_directory(
    '3 Class Images',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    '3 Class Images',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Building and compiling a CNN model, then fitting the data.
model = keras.Sequential([
	keras.layers.Conv2D(
        128,
        kernel_size=3,
        activation='relu',
        input_shape=(64, 64, 3)
	),
    keras.layers.MaxPool2D(4),
    keras.layers.Conv2D(
        128,
        kernel_size=3,
        activation='relu'
    ),
    keras.layers.MaxPool2D(3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
	keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit_generator(
    train_gen,
    epochs=20,
    validation_data=val_gen
)

# Graphing performance (accuracy) by number of epochs
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)


# Plotting loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.xticks(range(0, 21, 5))
plt.ylabel('Loss')
plt.legend()

plt.savefig('loss.png', dpi=500)
plt.show()

# Clearing the graph
plt.clf()

# Plotting accuracy
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.xticks(range(0, 21, 5))
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('acc.png', dpi=500)
plt.show()
