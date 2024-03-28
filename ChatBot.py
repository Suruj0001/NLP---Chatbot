#!/usr/bin/env python
# coding: utf-8

# In[1]:


##ChatBot App  ##PLAY


# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input , Embedding , LSTM , Dense , GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify


app = Flask(__name__)

# In[2]:


# %%writefile content.json
#  {
#   "intents": [
#     {
#       "tag": "greeting",
#       "input": [
#         "hello",
#         "hi there",
#         "nice to meet you",
#         "welcome to play",
#         "any one help"
#       ],
#       "responses": [
#         "Hello, Welcome to PLAY COMMUNITY",
#         "How may I help You",
#         "What Do You want",
#         "Glad You Visited"
#       ]
#     },
#     {
#       "tag": "Where is it located",
#       "input": [
#         "where is this Straw hat center located?",
#         "location where can I find it",
#         "What it does ? ",
#         "Who is the Fouder"
#       ],
#       "responses": [
#         "Up to you, USE MAPS , Near Guwhati , Assam ",
#         "PRESENTING YOU THE TECH HUB NATION",
#         "Defining Impossibility",
#         "Suruj Kalita"
#       ]
#     }
#   ]
# }


# In[3]:


# import json

# Open the 'content.json' file and load its content into 'data1'
with open('content.json') as content:
    data1 = json.load(content)


# In[4]:


##getting data to list
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
  responses[intent['tag']] = intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])


# In[5]:


##converting into dataframe
data = pd.DataFrame({"inputs":inputs,
                    "tags":tags})


# In[6]:


data


# In[7]:


data = data.sample(frac=1)


# In[7]:


#Preproccesing
##Before dealing with NLP CHATBOT Multiple Approcahes for Text #lstm #RNN #CNN
#removing punctuations
import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
data


# In[8]:


#Tokenizing
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(train)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])


# In[9]:


#Tensorlfow tokenizer for assigning unique token
input_shape = x_train.shape[1] ##parameter defined the calculation change it to get cleared of it
print(input_shape)


# In[10]:


##defining vocabulary and Unique Words
vocabulary = len(tokenizer.word_index)
print("Number of Unique Words: ", vocabulary)
output_length = le.classes_.shape[0]
print("output length " , output_length)


# In[11]:


#Artificial Neural Network
##LSTM #TRANSFORMER # RNN
#Creating the Model embedding the layers

i = Input(shape=(input_shape))

x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i,x)


# In[12]:


##compiling the Model

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


# In[13]:


##training the model
train = model.fit(x_train , y_train , epochs=200)


# In[14]:


##Model Analysis

plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'], label='training set loss')
plt.legend()


# In[16]:


# def predict_previous_behavior(input_message):
#     input_shape = tokenizer.text_to_sequences([input_message])
#     input_shape = pad_sequences(input_shape, maxlen=max_sequence_length, padding='post')
#     prediction_input = model.predict(input_shape)
#     # You can decode the predicted sequence back into text
#     predicted_text = tokenizer.sequences_to_texts([prediction_input.argmax(axis=-1)])
#     return prediction_input[0]


# In[15]:


#Testing

import random


def prediction():
  while True:
    text_p = []
    prediction_input = input('You : ')

    if prediction_input.lower() == "exit":
      print("Hope to See Soon")
      break

    ##removing punctuation
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    text_p.append(prediction_input)

    ##tokeinizing and padding
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    ##getting output shapes
    output = model.predict(prediction_input)
    output = output.argmax()

    # def predict_previous_behavior(input_message):
    #   input_shape = tokenizer.text_to_sequences([input_message])
    #   input_shape = pad_sequences(input_shape, maxlen=max_sequence_length, padding='post')
    #   prediction_input = model.predict(input_shape)
    #   # You can decode the predicted sequence back into text
    #   predicted_text = tokenizer.sequences_to_texts([prediction_input.argmax(axis=-1)])
    #   return prediction_input[0]

    ##finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    print("Protectron : ", random.choice(responses[response_tag]))
    if response_tag == "Will See You Again":
      break

prediction()


if __name__ == '__main__':
    app.run(debug=True)

# In[20]:


##left with front end Integration




# In[ ]:




