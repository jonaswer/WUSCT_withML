import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Download NLTK data files (only if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

def get_cleaned_data():
    """
    Reads and cleans the dataset.

    Returns:
        sentences (list): List of text sentences.
        ratings (np.ndarray): Array of adjusted ratings.
    """
    # Read data
    raw = pd.read_csv('AllData4noID.csv')
    raw['Text'] = raw['Text'].astype(str)

    # Remove duplicates
    cleaned = raw.drop_duplicates(subset=['Text'])

    # Extract sentences and ratings
    sentences = cleaned['Text'].values
    ratings = cleaned['Rating'].values.astype(int)

    # Adjust ratings to start from 0
    ratings = ratings - 2

    return sentences.tolist(), ratings

def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing, removing stopwords and punctuation, and stemming.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join words back into a single string
    return ' '.join(words)

def augment_data(X, Y):
    """
    Augments the data using SMOTE to handle class imbalance.

    Args:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Labels.

    Returns:
        X_augmented (np.ndarray): Augmented feature matrix.
        Y_augmented (np.ndarray): Augmented labels.
    """
    smote = SMOTE()
    X_augmented, Y_augmented = smote.fit_resample(X, Y)
    return X_augmented, Y_augmented

def compute_class_weights(Y):
    """
    Computes class weights to handle class imbalance during training.

    Args:
        Y (np.ndarray): Labels.

    Returns:
        dict: A dictionary mapping class indices to weights.
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(Y),
        y=Y
    )
    return dict(enumerate(class_weights))

def get_confusion_matrix(Y_true, X_test, model):
    """
    Generates the confusion matrix for the test set.

    Args:
        Y_true (np.ndarray): True labels (one-hot encoded).
        X_test (np.ndarray): Test feature matrix.
        model (tf.keras.Model): Trained model.

    Returns:
        np.ndarray: Confusion matrix.
    """
    # Convert one-hot encoded labels to class indices
    Y_true_indices = np.argmax(Y_true, axis=1)

    # Predict class probabilities
    Y_pred_probs = model.predict(X_test)
    # Convert predicted probabilities to class indices
    Y_pred_indices = np.argmax(Y_pred_probs, axis=1)

    # Compute confusion matrix
    cf_matrix = tf.math.confusion_matrix(Y_true_indices, Y_pred_indices)
    return cf_matrix.numpy()

def build_and_train_model():
    """
    Builds, trains, and evaluates the model.

    Returns:
        model (tf.keras.Model): Trained model.
        history (tf.keras.callbacks.History): Training history.
        tokenizer (Tokenizer): Fitted tokenizer.
    """
    sentences, ratings = get_cleaned_data()

    # Preprocess sentences
    sentences = [preprocess_text(sentence) for sentence in sentences]

    # Tokenizer parameters
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 100

    # Initialize and fit tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')

    # Convert sentences to sequences and pad them
    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    # Convert ratings to categorical (one-hot encoding)
    Y = ratings
    num_classes = np.max(Y) + 1
    Y = to_categorical(Y, num_classes=num_classes)

    # Augment data
    X, Y_indices = augment_data(X, np.argmax(Y, axis=1))
    Y = to_categorical(Y_indices, num_classes=num_classes)

    # Compute class weights
    class_weights = compute_class_weights(np.argmax(Y, axis=1))

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', Y.shape)

    # Build the model
    model = models.Sequential()
    model.add(layers.Embedding(
        input_dim=MAX_NB_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Training parameters
    epochs = 10
    batch_size = 50

    # Train the model
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f'Test accuracy: {test_acc:.4f}')

    # Compute confusion matrix
    cf_matrix = get_confusion_matrix(Y_test, X_test, model)
    print('Confusion Matrix:')
    print(cf_matrix)

    return model, history, tokenizer

def save_model(model, tokenizer):
    """
    Saves the trained model and tokenizer to disk.

    Args:
        model (tf.keras.Model): Trained model.
        tokenizer (Tokenizer): Fitted tokenizer.
    """
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save model
    model.save('trained_model.h5')

if __name__ == '__main__':
    model, history, tokenizer = build_and_train_model()
    save_model(model, tokenizer)
