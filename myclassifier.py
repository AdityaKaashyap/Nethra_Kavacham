import os
import numpy as np
import cv2
import pickle

data_dir = './data/diseases'
categories = ['catarct', 'conjuctivitis', 'glucoma', 'hyphema', 'irisMel', 'itits', 'keratocunus', 'normalEye', 'ptreygium', 'subconjuctival', 'uveti']
data = []

def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        if not os.path.exists(path):
            print(f"Directory {path} does not exist")
            continue

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))

                image = np.array(image, dtype=np.float32)
                data.append([image, label])
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue

    print(f"Total images processed: {len(data)}")
    with open('data.pickle', 'wb') as pik:
        pickle.dump(data, pik)

def load_data():
    with open('data.pickle', 'rb') as pick:
        data = pickle.load(pick)

    np.random.shuffle(data)
    features = []
    labels = []
    for img, label in data:
        features.append(img)
        labels.append(label)
    features = np.array(features, dtype=np.float32) / 255.0
    labels = np.array(labels)

    return features, labels

# Create the dataset
make_data()

# Example of how to use the load_data function
features, labels = load_data()
print(features.shape, labels.shape)
