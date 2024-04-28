import os
import numpy as np
import librosa
import pywt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
# Function to extract features from audio files
def extract_features(file_path, target_length):
    try:
        y, sr = librosa.load(file_path)

        # Noise reduction using wavelet transform
        coeffs = pywt.wavedec(y, 'db4', level=2)
        threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        y_reduced = pywt.waverec(coeffs, 'db4')

        mfccs = librosa.feature.mfcc(y=y_reduced, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y_reduced, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y_reduced, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y_reduced), sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y_reduced, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        features = np.vstack((mfccs, chroma, contrast, tonnetz, log_mel_spec))

        if features.shape[1] < target_length:
            features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])), mode='constant')
        else:
            features = features[:, :target_length]

        return features
    except Exception as e:
        print("Error in feature extraction:", e)
        return None

# Define a function to create the CNN model
# Define a CNN model function
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Folder path
data_folder = r"E:\Project\dataset\Check-20240221T085616Z-001\Check"

# Define emotions and folder combinations
emotions = ["Angry", "Neutral", "Happy", "Sad", "Disgust"]
folder_combinations = [
    (1, 2, 3, 4, 5),
    (1, 2, 3, 5, 4),
    (1, 2, 4, 5, 3),
    (1, 3, 4, 5, 2),
    (2, 3, 4, 5, 1)
]

# Initialize a label encoder
label_encoder = LabelEncoder()

# Initialize lists to store results
all_cm = []
all_reports = []
all_macro_f1 = []

# Iterate over each folder combination
for idx, folders_train_test in enumerate(folder_combinations):
    print(f"Training and testing using combination {idx+1}")

    # Extract features and labels
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Loop over each emotion and folder for training and testing
    for emotion in emotions:
        for folder in folders_train_test[0:4]:  # Use first four folders for training
            folder_name = f"{emotion}{folder}"
            folder_path = os.path.join(data_folder, folder_name)
            audio_files = os.listdir(folder_path)
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                features = extract_features(file_path, target_length=100)  # Target length can be adjusted
                if features is not None:
                    X_train.append(features)
                    y_train.append(emotion)

        for folder in folders_train_test[4:]:  # Use the fifth folder for testing
            folder_name = f"{emotion}{folder}"
            folder_path = os.path.join(data_folder, folder_name)
            audio_files = os.listdir(folder_path)
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                features = extract_features(file_path, target_length=100)  # Target length can be adjusted
                if features is not None:
                    X_test.append(features)
                    y_test.append(emotion)

    # Convert labels to categorical format
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Train the CNN model
    input_shape = X_train.shape[1:]
    num_classes = len(emotions)
    cnn_model = create_cnn_model(input_shape, num_classes)
    optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate
    cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)  # Increased patience
    history = cnn_model.fit(X_train, y_train_categorical, epochs=60, batch_size=64, validation_data=(X_test, y_test_categorical), callbacks=[early_stopping])

    # Evaluate the CNN model
    _, test_accuracy = cnn_model.evaluate(X_test, y_test_categorical, verbose=0)
    print("Test Accuracy:", test_accuracy)

    # Predict the labels for the test set
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_categorical, axis=1)

    # Calculate and print confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    all_cm.append(cm)

    # Calculate macro F1 score
    macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    print("Macro F1 Score:", macro_f1)
    all_macro_f1.append(macro_f1)

    # Print classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=emotions)
    print("Classification Report:")
    print(report)
    all_reports.append(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Print overall results
print("\nOverall Results:")
for idx, (cm, report, macro_f1) in enumerate(zip(all_cm, all_reports, all_macro_f1), 1):
    print(f"Combination {idx}:")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    print("Macro F1 Score:", macro_f1)
    print("\n")
