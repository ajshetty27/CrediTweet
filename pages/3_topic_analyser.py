import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from pymongo import MongoClient

st.set_page_config(page_title="Topic Analyser", page_icon="📈")
st.title("Topic Analyser")
st.sidebar.success("Topic Analyser")
st.sidebar.write("From the derived topics, take a quick look at a further analyses done on each topic in terms of the tweets that are related to the topic and the percentage of real and fake tweets for the topic.")

client = MongoClient("mongodb://localhost:27017");

print('request recieved')

mydatabase = client['main_dataset']

mycollection = mydatabase['data']

number = st.text_input("Input a topic number from -1 to 99", '-1')

topic = str(number)
counter = 0
counter_2 = 0
data_list = []
actual_count = 100
topic_name = " "
actual_count = '100'

for i in mydatabase.topics.find({'Topic': topic}):
    topic_name = i['CustomName']
    actual_count = i['Count']
st.header("Topic "+ "_:green["+topic_name+"]_")
for i in mydatabase.data.find({'Topic': topic}):
    data_list.append(i)

df = pd.DataFrame(data_list)
st.write("Total Tweets and data on topic = ",int(actual_count))
st.dataframe(df[['Text', 'Label']], use_container_width=True)

for i in mydatabase.data.find({'Topic': topic, 'Label':'REAL'}):
    counter+=1
for i in mydatabase.data.find({'Topic': topic, 'Label':'FAKE'}):
    counter_2+=1

col1, col2= st.columns(2)
col1.metric("_:red[Real Tweets]_" ,counter)
col2.metric("_:blue[Fake Tweets]_" ,counter_2)

if(counter/int(actual_count) >= counter_2/int(actual_count)):
    st.write("Topic of "+ topic_name+" is reliable as ", round(float(counter/int(actual_count)*100),2), "% of tweets are real")
else:
    st.write("Topic of "+ topic_name+" is NOT reliable as ", round(float(counter_2/int(actual_count)*100),2), "% of tweets are fake")

labels = 'Real', 'Fake'
colors = 'red', 'blue'
explode = [0.2,0]

sizes = [counter, counter_2]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors = colors, explode = explode, autopct='%1.1f%%', shadow = True, startangle=90)
ax1.axis('equal')

st.pyplot(fig1)

client.close();
