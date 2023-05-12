#import tweepy
#This code should provide a list of recent tweets from twiter within the dashboard to be later classified if an API key is provided 
import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from pymongo import MongoClient

st.set_page_config(page_title="Tweet Checks", page_icon="📈")
st.title("Tweet Checks ")
st.sidebar.success("Topic Analyser")
st.sidebar.write("You can view recent news tweets to check their credibility as well as the credibility scores for authors on twitter")


'''consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


api = tweepy.API(auth)


search_term = 'news'
tweet_count = 10


tweets = tweepy.Cursor(api.search_tweets,
                       q=search_term,
                       lang='en',
                       tweet_mode='extended').items(tweet_count)

# Print the recent news tweets
tweets_list = []

for tweet in tweets:
    print(tweet.full_text)
    tweets.append(tweet.full_text)

df_tweets = pd.DataFrame (tweets_list, columns = ['X_test'])'''
