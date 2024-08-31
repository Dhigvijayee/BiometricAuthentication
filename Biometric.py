import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def preprocess_image(image_path, is_binary=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    if is_binary:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image

def load_data(base_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Add other valid extensions if needed
    fingerprint_images = []
    left_iris_images = []
    right_iris_images = []
    labels = []

    for person_id in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_id)
        if os.path.isdir(person_dir):
            fingerprint_dir = os.path.join(person_dir, 'fingerprint')
            leftiris_dir = os.path.join(person_dir, 'leftiris')
            rightiris_dir = os.path.join(person_dir, 'rightiris')

            fingerprints = []
            left_irises = []
            right_irises = []

            for img_file in os.listdir(fingerprint_dir):
                if img_file.lower().endswith(valid_extensions):
                    img_path = os.path.join(fingerprint_dir, img_file)
                    image = preprocess_image(img_path, is_binary=True)
                    if image is not None:
                        fingerprints.append(image)

            for img_file in os.listdir(leftiris_dir):
                if img_file.lower().endswith(valid_extensions):
                    img_path = os.path.join(leftiris_dir, img_file)
                    image = preprocess_image(img_path)
                    if image is not None:
                        left_irises.append(image)

            for img_file in os.listdir(rightiris_dir):
                if img_file.lower().endswith(valid_extensions):
                    img_path = os.path.join(rightiris_dir, img_file)
                    image = preprocess_image(img_path)
                    if image is not None:
                        right_irises.append(image)

            
            min_count = min(len(fingerprints), len(left_irises) * 2, len(right_irises) * 2)
            fingerprints = fingerprints[:min_count]
            left_irises = (left_irises * (min_count // len(left_irises) + 1))[:min_count]
            right_irises = (right_irises * (min_count // len(right_irises) + 1))[:min_count]

            fingerprint_images.extend(fingerprints)
            left_iris_images.extend(left_irises)
            right_iris_images.extend(right_irises)
            labels.extend([person_id] * min_count)

    return (np.array(fingerprint_images),
            np.array(left_iris_images),
            np.array(right_iris_images),
            np.array(labels))


base_dir = '/kaggle/input/datasetset/finaldata/dataset'  # Adjust this path as necessary


fingerprint_images, left_iris_images, right_iris_images, labels = load_data(base_dir)


assert len(fingerprint_images) == len(left_iris_images) == len(right_iris_images) == len(labels), "Inconsistent dataset lengths"

fingerprint_images = fingerprint_images.reshape(-1, 128, 128, 1).astype('float32') / 255.0
left_iris_images = left_iris_images.reshape(-1, 128, 128, 1).astype('float32') / 255.0
right_iris_images = right_iris_images.reshape(-1, 128, 128, 1).astype('float32') / 255.0

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)


f_x_train, f_x_test, f_y_train, f_y_test = train_test_split(fingerprint_images, labels_categorical, test_size=0.2, random_state=42)
li_x_train, li_x_test, li_y_train, li_y_test = train_test_split(left_iris_images, labels_categorical, test_size=0.2, random_state=42)
ri_x_train, ri_x_test, ri_y_train, ri_y_test = train_test_split(right_iris_images, labels_categorical, test_size=0.2, random_state=42)


assert np.array_equal(f_y_train, li_y_train) and np.array_equal(f_y_train, ri_y_train), "Training labels must match across modalities"
assert np.array_equal(f_y_test, li_y_test) and np.array_equal(f_y_test, ri_y_test), "Test labels must match across modalities"

# Fingerprint model
fingerprint_input = Input(shape=(128, 128, 1))
x1 = Conv2D(32, (3, 3), activation='relu')(fingerprint_input)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Conv2D(64, (3, 3), activation='relu')(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Flatten()(x1)
x1 = Dense(128, activation='relu')(x1)

# Left iris model
left_iris_input = Input(shape=(128, 128, 1))
x2 = Conv2D(32, (3, 3), activation='relu')(left_iris_input)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)
x2 = Dense(128, activation='relu')(x2)

# Right iris model
right_iris_input = Input(shape=(128, 128, 1))
x3 = Conv2D(32, (3, 3), activation='relu')(right_iris_input)
x3 = MaxPooling2D((2, 2))(x3)
x3 = Conv2D(64, (3, 3), activation='relu')(x3)
x3 = MaxPooling2D((2, 2))(x3)
x3 = Flatten()(x3)
x3 = Dense(128, activation='relu')(x3)

# Combine all models
combined = Concatenate()([x1, x2, x3])
z = Dense(128, activation='relu')(combined)
z = Dense(len(label_encoder.classes_), activation='softmax')(z)


model = Model(inputs=[fingerprint_input, left_iris_input, right_iris_input], outputs=z)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit([f_x_train, li_x_train, ri_x_train], f_y_train, epochs=20, batch_size=32, validation_data=([f_x_test, li_x_test, ri_x_test], f_y_test))


loss, accuracy = model.evaluate([f_x_test, li_x_test, ri_x_test], f_y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


model.save('multi_biometric_recognition_model.h5')