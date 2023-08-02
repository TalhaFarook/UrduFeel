import pandas
import numpy 
pandas.options.mode.chained_assignment = None
pandas.set_option('display.max_colwidth', None)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# !pip install keras
# !pip install tensorflow
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

# Load the model data from the pickle file
with open('urduSentiModel', 'rb') as file:
    modelData = pickle.load(file)

# Extract the components from the loaded model data
tokens = modelData['tokenizer']
maxLength = modelData['maxLength']
model = modelData['model']

def makePrediction(sentence):
    # Preprocess the input sentence
    sentenceSequence = tokens.texts_to_sequences([sentence])
    sentencePad = pad_sequences(sentenceSequence, maxlen = maxLength)

    # Make sentiment prediction using the loaded model
    prediction = model.predict(sentencePad)
    predictedClass = round(prediction[0][0])

    # Convert the prediction to 'P' (Positive) or 'N' (Negative)
    sentiment = 'P' if predictedClass == 1 else 'N'
    
    return sentiment

# Write your sentence here
sentence = 'چندے سے انقلاب اور عمران خان وزیر اعظم نہیں بن سکتے'
print(f"Predicted Sentiment: {makePrediction(sentence)}")