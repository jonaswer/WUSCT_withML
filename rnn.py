import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import Flatten, Dense  
from keras.layers import Embedding
from keras.layers import SpatialDropout1D, LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# read and clean the dataset
def get_cleaned_data():

    # read data
    raw = pd.read_csv('AllData4noID.csv')  
    raw.Text=raw.Text.astype(str)

    # clean data from dublicates
    dublicate_cleaned = raw.drop_duplicates(subset=['Text'])
  
    # create numpy arrays
    sentences = dublicate_cleaned['Text'].values
    ratings = dublicate_cleaned['Rating'].values

    # fit labels beginning from 0 
    for i in range(len(ratings)):
        ratings[i] = ratings[i]-2 

    return list(sentences), ratings


# some preprocessing for text based data
def get_tokenizer_preprocesing(text):
        # Convert the text to lowercase
        text = text.lower()

        # Tokenize the text into words
        words = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Remove punctuation
        words = [word for word in words if word.isalpha()]

        # Stem the words (optional)
        stemmer = nltk.PorterStemmer()
        words = [stemmer.stem(word) for word in words]

        # Join the words back into a single string
        text = " ".join(words)

        return text


# augment the data
def get_data_augmented(X,Y):

    smote = SMOTE()
    X_augmented, y_augmented = smote.fit_resample(X, Y)
    X = X_augmented
    Y = y_augmented
    return X,Y


# Calculate the class weights based on the frequency of each class in the training data
def get_class_weights(Y):
   
    class_weights = compute_class_weight(    
                                        class_weight = "balanced",
                                        classes = np.unique(Y),
                                        y = Y                                                 
                                    )
    class_weights = dict(zip(np.unique(Y), class_weights))

    return class_weights


# binary ratings for one-hot encoding, due to a small dataset
def get_one_hot_encoding(Y):

    ratings_2d = []
    for i in range(len(Y)):
        rating = []
        for j in range(7):
            if Y[i] == j: 
                rating.append(1)
            else:
                rating.append(0)
        ratings_2d.append(rating)
    ratings = np.array(ratings_2d)   
    Y = ratings  

    return Y


# print confusion matrix
def get_confusion_matrix(Y_test, X_test, model):

    # print confusion matrix
    confusion_true = []
    for key, value in enumerate(Y_test):
        binary_list = value
        max_item = max(binary_list)
        index_max_item = [index for index, item in enumerate(binary_list) if item == max_item][0]
        confusion_true.append(index_max_item)

    test_set = X_test
    prediction_list = model.predict(test_set)
    confusion_predictions = []

    for key, value in enumerate(prediction_list):
        binary_list = value
        max_item = max(binary_list)
        index_max_item = [index for index, item in enumerate(binary_list) if item == max_item][0]
        confusion_predictions.append(index_max_item)

    cf_matrix = tf.math.confusion_matrix(confusion_true, confusion_predictions)

    return cf_matrix
  

# read the raw data, clean it with help of function above and train a model
def get_model():

    sentences, ratings = get_cleaned_data()

    # preprocessin for tokenizer
    for i in range(len(sentences)):
        sentences[i] = get_tokenizer_preprocesing(sentences[i])
    sentences = np.asarray(sentences)


    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each sentecne completion
    MAX_SEQUENCE_LENGTH = 50
    # This is fixed.
    EMBEDDING_DIM = 100

    # Tokenizer to create a feature set
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # X/Y, answers/ratings
    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = ratings
    
    # data augmentation 
    X,Y = get_data_augmented(X,Y)
  
    class_weights = get_class_weights(Y)
  
    # binary ratings for one-hot encoding, due to a small dataset
    Y = get_one_hot_encoding(Y)

    # Tensor shapes
    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', Y.shape)

    # train test split with 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)      

    ## Architecture of RNN network

    # Define the input layer
    MAX_SEQUENCE_LENGTH = X.shape[1]
    inputs = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Embed the input layer
    x = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inputs)

    # Add a bidirectional LSTM layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)

    # Add a dropout layer
    x = tf.keras.layers.Dropout(0.5)(x)

    # Add a dense layer with 'num_classes' units and a softmax activation function
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])#[tf.keras.metrics.CategoricalAccuracy()])#metrics=['accuracy'])

    # training parameter 
    epochs = 1
    batch_size = 50

    # model training class_weight=class_weights
    history = model.fit(X_train, Y_train, epochs=epochs, class_weight=class_weights, batch_size=batch_size,validation_split=0.5,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    ## Testing
    # print accuracy
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)

    # print confusion matrix
    cf_matrix = get_confusion_matrix(Y_test, X_test, model)
    print(cf_matrix)
   
    
    return model, history, tokenizer
    

# save a trained model
def save_model(model, tokenizer):

    # save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save neural network weights
    model.save('/home/jonas/Desktop/priv_proj/wusct/new')


if __name__ == "__main__":

    model, history, tokenizer = get_model()
    save_model(model, tokenizer)
