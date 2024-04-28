import os
import numpy as np
import librosa
import pywt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score

# Function to extract features
def extract_features(audio_file_path, target_length):
    try:
        y, sr = librosa.load(audio_file_path)

        # Noise reduction using wavelet transform
        coeffs = pywt.wavedec(y, 'db4', level=2)
        threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        y_reduced = pywt.waverec(coeffs, 'db4')

        mel_spec = librosa.feature.melspectrogram(y=y_reduced, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize log mel spectrogram to the input size of VGG16
        resized_log_mel_spec = np.resize(log_mel_spec, (224, 224))

        return resized_log_mel_spec
    except Exception as e:
        print("Error in feature extraction:", e)
        return None

# Load data and extract features
data_folder = r"E:\Project\dataset\Check-20240221T085616Z-001\Check"
emotions = ["Angry", "Neutral", "Happy", "Sad", "Disgust"]
folder_combinations = [
    [(1, 2, 3, 4), (5,)],
    [(2, 3, 4, 5), (1,)],
    [(1, 4, 5, 2), (3,)],
    [(3, 4, 5, 1), (2,)],
    [(1, 3, 5, 2), (4,)]
]
target_length = 300

# Load pre-trained VGG16 model without classification layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for i, (train_folders, test_folders) in enumerate(folder_combinations, 1):
    print(f"Set {i}:")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for emotion in emotions:
        for folder in train_folders:
            folder_name = f"{emotion}{folder}"
            folder_path = os.path.join(data_folder, folder_name)
            audio_files = os.listdir(folder_path)
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                features = extract_features(file_path, target_length=target_length)
                if features is not None:
                    X_train.append(features)
                    y_train.append(emotion)

        for folder in test_folders:
            folder_name = f"{emotion}{folder}"
            folder_path = os.path.join(data_folder, folder_name)
            audio_files = os.listdir(folder_path)
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                features = extract_features(file_path, target_length=target_length)
                if features is not None:
                    X_test.append(features)
                    y_test.append(emotion)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Preprocess input data to have 3 channels (as required by VGG16)
    X_train = np.stack((X_train,) * 3, axis=-1)
    X_test = np.stack((X_test,) * 3, axis=-1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    # Flatten features to be compatible with the fully connected layers of VGG16
    X_train_flatten = X_train.reshape((X_train.shape[0], 224 * 224 * 3))
    X_test_flatten = X_test.reshape((X_test.shape[0], 224 * 224 * 3))

    # Add fully connected layers for classification
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(emotions), activation='softmax'))

    # Freeze the layers of the pre-trained VGG16 model
    vgg_model.trainable = False

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_categorical, epochs=50, batch_size=64, validation_data=(X_test, y_test_categorical), verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
    print("Test Accuracy:", test_accuracy)
