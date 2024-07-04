from utils import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
(features, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

# Define categories and number of classes
categories = ['cataract', 'conjuctivitis', 'glucoma', 'hyphema', 'irisMel', 'itits', 'keratocunus', 'normalEye', 'ptreygium', 'subconjuctival', 'uveti']
num_classes = len(categories)

# Define the model architecture
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

flat1 = tf.keras.layers.Flatten()(pool4)
dense1 = tf.keras.layers.Dense(512, activation='relu')(flat1)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(dense1)

# Create the model
model = tf.keras.Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Save the model
model.save('mymodel.h5')

# Plot training history
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
