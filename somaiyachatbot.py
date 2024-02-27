import nltk
import random
import numpy as np
import string
import tkinter as tk
from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from PIL import ImageTk, Image

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

file = open("C:/Users/Mukesh/OneDrive/Desktop/project 2/final chatbot/sksc.txt", "r", errors="ignore")
sksc_doc = file.read().lower()

sentence_tokens = nltk.sent_tokenize(sksc_doc)
word_tokens = nltk.word_tokenize(sksc_doc)

lemm = nltk.stem.WordNetLemmatizer()

def lemToken(tokens):
    return [lemm.lemmatize(token) for token in tokens]

stop_words = set(stopwords.words('english'))

def lemNormalize(text):
    return lemToken([word for word in nltk.word_tokenize(text.lower().translate(remove_punc_dict)) if word not in stop_words])

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

greet_input = ('hello', 'hi', 'hii', 'sup', 'whassup', '...', '?', 'how are you?')
greet_response = ('ho', 'heyy', 'hey there!', 'hello', 'hello', 'hello', 'hello')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_input:
            return random.choice(greet_response)

def response(user_response):
    bot_response = " "
    Tfi = TfidfVectorizer(tokenizer=lemNormalize, stop_words="english")
    tf = Tfi.fit_transform(sentence_tokens)
    val = cosine_similarity(tf[-1], tf)
    indfex = val.argsort()[0][-2]
    flat = val.flatten()
    flat.sort()
    req_tf = flat[-2]
    if req_tf == 0:
        bot_response = bot_response + "Sorry, I did not understand your question."
        return bot_response
    else:
        bot_response = bot_response + sentence_tokens[indfex]
        return bot_response

from tkinter import messagebox

def send(event=None):
    global sentence_tokens, word_tokens
    
    user_response = entry_field.get()
    user_response = user_response.lower()
    entry_field.delete(0, END) # Clear the entry field
    if user_response != 'bye':
        if user_response == "thank you" or user_response == "thanks" or user_response == "thank you so much":
            response_area.insert(END, 'Bot: You are welcome!\n')
        else:
            if greet(user_response) != None:
                response_area.insert(END, 'Bot: ' + greet(user_response) + '\n')
            else:
                sentence_tokens.append(user_response)
                word_tokens = nltk.word_tokenize(sksc_doc.lower())
                final_words = list(set(word_tokens))
                
                # Display user message
                response_area.configure(state=NORMAL)
                response_area.insert(END, 'You: ' + user_response + '\n')
                
                # Get and display bot response
                bot_response = response(user_response)
                response_area.insert(END, 'Bot: ' + bot_response + '\n')
                
                response_area.configure(state=DISABLED)
    else:
        response_area.insert(END, 'Bot: Goodbye! Take care.\n')
        root.destroy()
    response_area.yview(END) # Scroll to the bottom of the chat history

# create the GUI
root = Tk()

#set the root title and size
root.title('Chatbot')
root.geometry("600x500")

# Load the logo image
logo_image = ImageTk.PhotoImage(Image.open("C:/Users/Mukesh/Downloads/bot1.png"))

# Create a label with the logo image
logo_label = Label(root, image=logo_image)
logo_label.pack()

# Create a label for the chat history
history_label = Label(root, text="ChatBoT", font="Helvetica 16 bold")
history_label.pack(pady=10)

#Create a text area for displaying the chat history and make it read-only
response_area = Text(root, height=10, width=50, font="Helvetica 10", state=DISABLED)
response_area.pack(side=TOP, fill=BOTH, expand=True)
response_area.pack(padx=10, pady=10)
response_area.configure(state=NORMAL)
response_area.insert(END, "Bot: Welcome! How can I help you today?\n")
response_area.configure(state=DISABLED)

#Create a label and an entry field for the user to input text
entry_label = Label(root, text="Enter your message here:", font="Helvetica 12 bold")
entry_label.pack(pady=10)

# create the scrollbar
scrollbar = Scrollbar(response_area)
scrollbar.pack(side=RIGHT, fill=Y)
response_area.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=response_area.yview)

# create the entry field and send button
entry_field = Entry(root)
entry_field.bind("<Return>", send)
entry_field.pack(side=LEFT, fill=X, expand=True)
send_button = Button(root, text='Send', command=send)
send_button.pack(side=RIGHT)

# focus on the entry field
entry_field.focus()

# start the GUI
root.mainloop()

       
