import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import time
import matplotlib.pyplot as plt
from pymongo import MongoClient

#Set page configurations
st.set_page_config(page_title="Accounts", page_icon="📖")
st.title("Accounts")
st.sidebar.success("Accounts")
st.sidebar.write("View the credibility of different Accounts found in tweets and datasets")
#Load in MongoDB client for Pymongo
client = MongoClient("mongodb://localhost:27017");

print('request recieved')

mydatabase = client['main_dataset']

mycollection = mydatabase['data']

temp_list = []
accountNames = []
credibilities = []
account_numbers = []

progress_text = "Generating credibilities for Accounts"
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

for i in mydatabase.accounts.find():
    account = i["Account"]
    accountNames.append(account)
    credibility = int(i["Credibility"])
    credibilities.append(credibility)
    account_numbers.append(i)


st.subheader("_:green[Sort by]_")

df = pd.DataFrame({"Account Name": accountNames, "Credibility(%)": credibilities})
sorted_df = df

col1, col2, col3= st.columns(3)


with col1:
    if st.button('By Account'):
       sorted_df = df

with col2:
    if st.button('In ascending order'):
        sorted_df = df.sort_values(by = "Credibility(%)")

with col3:
    if st.button('In descending order'):
       sorted_df = df.sort_values(by = "Credibility(%)", ascending=False)

st.dataframe(sorted_df, use_container_width=True)

chart_data = pd.DataFrame({"Accounts": account_numbers, "Credibility(%)": credibilities})

st.bar_chart(chart_data, y="Credibility(%)")
