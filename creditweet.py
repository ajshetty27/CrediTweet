#Main Page for CrediTweet, use command line streamlit run creditweet.py to run the dashboard on your device
import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle5 as pickle
from twitter_check import predictor
import pymongo
import matplotlib.pyplot as plt
from pymongo import MongoClient
import time

st.set_page_config(
    page_title="CrediTweet",
    page_icon="🐦",
)

progress_text = "Loading CrediTweet"
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.2)
    my_bar.progress(percent_complete + 1, text=progress_text)


client = MongoClient("mongodb://localhost:27017");

print('request recieved')

mydatabase = client['main_dataset']

mycollection = mydatabase['data']



st.title("Welcome to Creditweet, your tweet credibility checker")

X_test = st.text_input("Copy and paste your tweet here to check its credibility", 'OBAMA REGIME’S SECRET ASIAN TRADE DEAL Would Let International Tribunal Overrule State and Fed Laws To Benefit Foreign Companies')

if X_test != " ":
    #Predictor is function called from twitter_check code, that python file contains all the predictor functions for models used
    label, confidence_score, num_of_topics, similar_topics, similarity_rounded, new_text_top_words = predictor(X_test)

    print(label)

    colour = " "

    if label == "REAL":
        colour = "_:red"
    else:
        colour = "_:blue"

    st.write("The tweet displayed is: "+colour+"["+label+"]_")
    st.write("The confidence score for the displayed tweet (Fake<0<Real):", float(confidence_score))
    st.write("Significant words in the provided text that result in the text being: " +colour+"["+label+"]_", new_text_top_words)

    print(similar_topics[0])
    assigned_topic = similar_topics[0]

    for i in mydatabase.topics.find({'Topic': str(assigned_topic)}):
        topic_name = i['CustomName']
        print(topic_name)

    st.write("A similar topic to the tweet above is topic "+str(assigned_topic)+": "+"_:green["+topic_name+"]_")
    st.write("Similarity score: ", int(similarity_rounded))
    st.write("View similar tweets that are real")

    dl = []

    for i in mydatabase.data.find({'Topic': str(assigned_topic), 'Label': "REAL"}):
        dl.append(i)

    dfl = pd.DataFrame(dl)
    st.dataframe(dfl[['Text', 'Label']], use_container_width=True)


st.write("Model operating at 84.6% accuracy, please be advised when using this tool.")

st.sidebar.success("Explore topics and accounts in more detail here")
