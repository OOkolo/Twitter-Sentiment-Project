## Imports Section
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
pd.set_option("display.max_colwidth", -1)
import numpy as np
import pandas as pd
import tweepy
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud
import openpyxl
import time
import tqdm
import time

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#==========================================================================================================================================

#Main function
def main():

  #Set header and subheader
  st.title("Live Twitter Sentiment Analysis")
  st.subheader("Enter a topic and find out what people are saying about it: ")

  html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment Analysis</p></div>
	"""

  st.markdown(html_temp, unsafe_allow_html=True)
  st.subheader("Select a topic which you'd like to get the sentiment analysis on :")


  #=========================================================================================================================================
  #TWITTER AUTHENTIFICATION
  #========================
  consumer_key = "2n1KPD1TwAmTBXgw0GRjsoswq"
  consumer_secret = "SqCsm7asKLho3pebXjrmLsl0Gyllo3GXFNNShA8IR4rVKIHRBw" #consumer secret
  access_token= "3394460153-7iYrs2SOG3A7cpFsGnKp7bjXOxQeWoXiZSWudN4"
  access_secret= "Nw1xVT1O8dKwRKVwmoLXhnP4IDBQ954FaLs4UYF27LHF3" #access secret

  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)

  api = tweepy.API(auth)


  #=========================================================================================================================================
  #Building the dataframe
  #====================== 
  tweets_df = pd.DataFrame(columns=["date","user","is_verified","tweet","likes","RT",'user_location'])



  #Function to extract tweets
  #========================== 
  def get_tweets(topic, count):
    i=0
    for tweet in tweepy.Cursor(api.search, q = topic, count = 100, lang="en",exclude='retweets').items():
      print(i, end= "\r")
      tweets_df.loc[i, "date"] = tweet.created_at
      tweets_df.loc[i, "user"]= tweet.user.name
      tweets_df.loc[i, "is_verified"] = tweet.user.verified
      tweets_df.loc[i, "tweet"]= tweet.text
      tweets_df.loc[i, "likes"] = tweet.retweet_count
      tweets_df.loc[i, "user_location"] = tweet.user.location
      tweets_df.to_csv("/content/drive/MyDrive/Colab Notebooks/Twitter Sentiment Analysis/TweetDataset.csv",index=False)
      tweets_df.to_excel('{}.xlsx'.format("/content/drive/MyDrive/Colab Notebooks/Twitter Sentiment Analysis/TweetDataset"),index=False)   ## Save as Excel
      i=i+1
      if i> count:
        break
      else:
        pass
    
  
  #Function to clean tweets
  #=========================
  def clean_tweet(tweet):
    return " ".join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())
  

  #Function to analyze sentiment
  #=============================
  def analyse_sentiment(tweet):
    #Make textblob object
    analysis = TextBlob(tweet)
    #positive sentiment condition
    if analysis.sentiment.polarity > 0:
      return "Positive"
    #Neutral sentiment condition
    if analysis.sentiment.polarity == 0:
      return "Neutral"
    #Negative condition
    if analysis.sentiment.polarity < 0:
      return "Negative"

  

  #Function to preprocess the data to remove stopwords and topic text
  #==================================================================
  def remove_stopwords(message):
    removed = stopwords.words("english")
    new_message = " ".join([word for word in message.split() if word.lower() not in removed])
    new_message = new_message.replace(Topic[0].lower(), "")
    return new_message




  #Display the image for the logo
  #==============================
  
  from PIL import Image
  image = Image.open('/content/drive/MyDrive/Colab Notebooks/Twitter Sentiment Analysis/polar-blind-date.png')
  st.image(image, caption='Twitter for Analytics',use_column_width=True)



  #Collect Input from the User
  #===========================
  Topic = str()
  Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))

  if len(Topic) > 0:

    #Call function to extract data on the topic
    with st.spinner("We're finding all the tweets about {}".format(Topic)):
      get_tweets(Topic , count=200)
    time.sleep(3)
    st.success('Good news... We found them!') 

    #Call function to clean tweet data
    tweets_df['clean_tweet'] = tweets_df["tweet"].apply(clean_tweet)

    #Call function to find the sentiment
    tweets_df["sentiment"] = tweets_df['clean_tweet'].apply(analyse_sentiment)


    # Write Summary of the Tweets
    st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(tweets_df.Tweet)))
    st.write("Total Positive Tweets are : {}".format(len(tweets_df[tweets_df["sentiment"]=="Positive"])))
    st.write("Total Negative Tweets are : {}".format(len(tweets_df[tweets_df["sentiment"]=="Negative"])))
    st.write("Total Neutral Tweets are : {}".format(len(tweets_df[tweets_df["sentiment"]=="Neutral"])))


    #Procedure for creating buttons in streamlit:
    #use the "if st.button('Button text')" syntax because it creates the button 
    #using the st.button syntax, the "if" statement is true or false depending
    #on the button's click state






    #See the extracted data
    if st.button("View the Extracted Data"):
      st.markdown(html_temp, unsafe_allow_html=True)
      st.success("Below is the Extracted Data :")
      st.write(tweets_df.head(50))

    #Generate pyplot
    if st.button("Get Countplot for Different Sentiments"):
      st.success("Generating A Count Plot")
      st.subheader(" Count Plot for Different Sentiments")
      st.write(sns.countplot(tweets_df["sentiment"]))
      st.pyplot()
    
    #Piechart
    if st.button("Get Pie Chart for Different Sentiments"):
      st.success("Generating A Pie Chart")
      a=len(tweets_df[tweets_df["sentiment"]=="Positive"])
      b=len(tweets_df[tweets_df["sentiment"]=="Positive"])
      c=len(tweets_df[tweets_df["sentiment"]=="Positive"])
      d=np.array([a,b,c])
      explode = (0.1, 0.0, 0.1)
      st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%'))
      st.pyplot()

    #Countplot for unverified users vs verified
    if st.button("Get Bar Plot Based on Verified and unverified Users"):
      st.success("Generating a Bar Chart")
      #chart title
      st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
      st.write(sns.countplot(tweets_df["sentiment"],hue=tweets_df.is_verified))
      st.pyplot()
    

    #Create a WordCloud
    if st.button("Get WordCloud for all things said about {}".format(Topic)):
      st.success("Generating A WordCloud for all things said about {}".format(Topic))
      text = " ".join(review for review in tweets_df.clean_tweet)
      text_new = remove_stopwords(text)
      wordcloud = WordCloud(max_words=800,max_font_size=70).generate(text_new)
      st.write(plt.imshow(wordcloud))
      st.pyplot()
    

    #Create a WordCloud for Positive tweets
    if st.button("Get WordCloud for all Positive things said about {}".format(Topic)):
      st.success("Generating A WordCloud for all things said about {}".format(Topic))
      text = " ".join(review for review in tweets_df[tweets_df["sentiment"]== "Positive"]["clean_tweet"])
      text_new = remove_stopwords(text)
      wordcloud = WordCloud(max_words=800,max_font_size=70).generate(text_new)
      st.write(plt.imshow(wordcloud))
      st.pyplot()


    #Create a WordCloud for Negative tweets
    if st.button("Get WordCloud for all Negative things said about {}".format(Topic)):
      st.success("Generating A WordCloud for all things said about {}".format(Topic))
      text = " ".join(review for review in tweets_df[tweets_df["sentiment"]== "Negative"]["clean_tweet"])
      text_new = remove_stopwords(text)
      wordcloud = WordCloud(max_words=800,max_font_size=70).generate(text_new)
      st.write(plt.imshow(wordcloud))
      st.pyplot()

  


  #=================================================================================================================
  #Building the Sidebar
  #====================
  st.sidebar.header("About App")
  st.sidebar.text("Made by Olisa Okolo 🔆 with Streamlit")
  st.sidebar.info("A Twitter visualisation project that performs a live analysis of the sentiments that Twitter users hold about your desired topic. \nThe visualisations are useful for determining the general sentiment of a topic at the time of search. \nOne can also view the most common words used when positively or negatively discussing the desired topic  ")

  st.sidebar.header("For Any questions/Suggestions Please reach out at :")
  st.sidebar.info("ookolo1@umbc.edu")



  #=================================================================================================================
  #Exit Strategy
  #=============
  if st.button("Exit"):
    st.balloon()


if __name__ == '__main__':
    main()
