import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Define constants
batch_size = 32
epochs = 20
img_height, img_width = 224, 224  # Adjust these dimensions based on your images

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')  # Adjust num_classes based on your dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Define image data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split for validation
)

# Absolute path to the data directory
data_dir = r'C:\Users\Lenovo\OneDrive\Desktop\HAckathon\data\diseases'

# Print the absolute path to verify
print(f"Data directory: {data_dir}")

# Ensure the directory exists
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exist.")
else:
    print(f"Directory {data_dir} exists.")

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    data_dir,  # Use absolute path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Flow validation images in batches using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,  # Use absolute path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
x_batch, y_batch = next(train_generator)
print(f"Train batch X shape: {x_batch.shape}")
print(f"Train batch Y shape: {y_batch.shape}")

x_val_batch, y_val_batch = next(validation_generator)
print(f"Validation batch X shape: {x_val_batch.shape}")
print(f"Validation batch Y shape: {y_val_batch.shape}")

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1
)

# Evaluate on validation set
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy * 100:.2f}%')

# Example of how to use the model for prediction
# Load and preprocess the image
img_path = r'C:\Users\Lenovo\OneDrive\Desktop\HAckathon\data\diseases\hyphema\Fig1-external-hyphema-OS.png' 
img = load_img(img_path, target_size=(img_height, img_width))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

# Make predictions
predictions = model.predict(x)
predicted_class = np.argmax(predictions)

# Print the predicted class
print(f'Predicted class: {predicted_class}')
