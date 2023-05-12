import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import time
import matplotlib.pyplot as plt
from pymongo import MongoClient

st.set_page_config(page_title="Topics", page_icon="📖")
st.title("Topics")
st.sidebar.success("Topics")
st.sidebar.write("View the credibility of different topics found in tweets and datasets. By using Bertopic, we can determine unique topics from tweets, and, with the use of the previous BERT model we can determine the credibility of different topics currently on the internet.")

client = MongoClient("mongodb://localhost:27017");

print('request recieved')

mydatabase = client['main_dataset']

mycollection = mydatabase['data']

topicNo = []
topicNames = []
credbilities = []

progress_text = "Generating credibilities for topics"
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

for i in mydatabase.topics.find():
    counter = 0
    topic = i['Topic']
    topic_name = i['CustomName']
    actual_count = i['Count']
    for i in mydatabase.data.find({'Topic': topic, 'Label':'REAL'}):
        counter+=1
    credbility = round(float(counter/int(actual_count)*100),2)
    topicNo.append(topic)
    topicNames.append(topic_name)
    credbilities.append(credbility)

st.subheader("_:green[Sort by]_")

df = pd.DataFrame({"Topic No.": topicNo, "Topics": topicNames, "Credibility(%)": credbilities})
sorted_df = df

col1, col2, col3= st.columns(3)


with col1:
    if st.button('By Topic'):
       sorted_df = df

with col2:
    if st.button('In ascending order'):
        sorted_df = df.sort_values(by = "Credibility(%)")

with col3:
    if st.button('In descending order'):
       sorted_df = df.sort_values(by = "Credibility(%)", ascending=False)

st.dataframe(sorted_df, use_container_width=True)

chart_data = pd.DataFrame({"Topic No.": topicNo, "Credibility(%)": credbilities})

st.bar_chart(chart_data, y="Credibility(%)")
