#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')


# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
# words = all words, vocabulary
#print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype='object')
# create train and test lists. X - patterns, Y - intents
x = list(training[:,0])
y = list(training[:,1])

train_x = x[:95]
train_y = y[:95]

test_x = x[5:]
test_y = y[5:]
print("Training set and Testing set created")

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Create model - 3 layers. First layer 250 neurons, second layer 125 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(250, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, verbose=1, batch_size=10, validation_data=(np.array(test_x), np.array(test_y)))
loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print("Testing Accuracy: {:.4f}".format(accuracy))
model.save('chatbot_model.h5', hist)

model.predict(test_x)

print("Model created")
import numpy
import matplotlib.pyplot as plt
from keras.utils import plot_model

fig = plt.figure(figsize=(10,4))
plt.plot(hist.history["accuracy"],color="blue",label="Training Accuracy")
plt.plot(hist.history["val_accuracy"],color="red",label="Testing Accuracy")
plt.title("Model accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.show()


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import csv
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = msg_entry.get()
    msg_entry.delete(0,END)

    if msg != '':
        text_widget.config(state=NORMAL)
        text_widget.insert(END, "YOU : " + msg + '\n\n')
        text_widget.config(foreground="white", font=(FONT_BOLD, 15 ))
    
        res = chatbot_response(msg)
        text_widget.insert(END, "BOT : " + res + '\n\n')
            
        text_widget.config(state=DISABLED)
        text_widget.yview(END)
 

base = Tk()
base.title("JARVIS HOTEL CHATBOT")
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
#base._setup_main_window()
base.resizable(width=False, height=False)
base.configure(width=470, height=550, bg=BG_COLOR)

#head label
head_label = Label(base, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10)
head_label.place(relwidth=1)

#line label
line=Label(base, width=450, bg=BG_GRAY)
line.place(relwidth=1, rely=0.07, relheight=0.012)

text_widget = Text(base, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
text_widget.configure(cursor="arrow", state=DISABLED)

#scroll bar
#scrollbar = Scrollbar(text_widget)
#scrollbar.place(relwidth=1, relx=0.99)
#scrollbar.pack(side=RIGHT, fill=Y, pady=0.5)
#scrollbar.configure(command=text_widget.yview, cursor="arrow")
#text_widget['yscrollcommand'] = scrollbar.set

#bottom label
bottom_label = Label(base, bg=BG_GRAY, height=80)
bottom_label.place(relwidth=1, rely=0.825)

#message entry box
msg_entry = Entry(bottom_label, bg="#2c3e50", fg=TEXT_COLOR, font=FONT)
msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)

#button
SendButton = Button(bottom_label, text="SEND", font=FONT_BOLD, width=20, bg=BG_GRAY, command = send)
SendButton.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

base.mainloop()





