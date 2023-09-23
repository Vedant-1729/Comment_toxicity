
# pip install --upgrade pip

# # !pip install tensorflow tensorflow-gpu pandas matplotlib sklearn
# pip install tensorflow
# # pip install tensorflow-gpu
# pip install pandas
# pip install matplotlib
# pip install scikit-learn

import os
import pandas as pd
import tensorflow as tf
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

path1 = '/content/drive/MyDrive/Colab Notebooks/Dataset/comment_toxicity/train.csv/train.csv'
df= pd.read_csv(path1, encoding='ISO-8859-1')
df

"""Text vectorization convert text data into a numerical format that can be used as input to machine learning models.
In Text Vectorization first step is tokenization,then vocab creation and then vectorization like word embedding
"""
from tensorflow import keras
#from tensorflow.keras.layers import TextVectorization
from keras import layers
from keras.layers import TextVectorization

X = df['comment_text']
# In Y we store the features i.e we take col after comment_text and .value convert it into numpy array.
y = df[df.columns[2:]].values

# Output_sequence length specifies the max length of sentence we take from comment_text section
# and the output_Mode convert convert each word and asssign it to no.
MAX_FEATURES = 200000 # number of words in the vocab
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(X.values)

# example of vectorizer
vectorizer('My Name is Vedant Shinde')[:5]

vectorized_text = vectorizer(X.values)
vectorized_text

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
# creating a tenserflow data pipeline
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks

# we assign 70% of our dataset to train 20% to validation and 10% to test
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

"""Creating Sequential Model"""


# from tensorflow.keras.models import Sequential
from keras import models
from keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

model = Sequential()
# Create the embedding layer
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer we are doing 6 difeerent classifier therefore in dense layer we have 6 nodes
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')

model.summary()

# run it for more epochs and add dropout layer of 0.3 to increase the accuracy
hist = model.fit(train, epochs=1, validation_data=val)

"""Make Predictions"""

input_text = vectorizer('You freaking suck!')
input_text.shape

batch_X, batch_y = test.as_numpy_iterator().next()

# res = model.predict(np.expand_dims(input_text,0))
# res
# (res > 0.5).astype(int)

(model.predict(batch_X) > 0.5).astype(int)

(model.predict(batch_y) > 0.5).astype(int)

"""Evaluate the model"""

# from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.metrics import Precision, Recall,CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    # Make a prediction
    yhat = model.predict(X_true)

    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
# the accuracy is less as we train the model for one epoch only

"""Use Gradio App for UI"""

# !pip install gradio jinja2

import tensorflow as tf
import gradio as gr

model.save('comment_toxicity.h5')

model = tf.keras.models.load_model('comment_toxicity.h5')

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)

    return text

interface = gr.Interface(fn=score_comment,inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)