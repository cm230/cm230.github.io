# Databricks notebook source
# MAGIC %md # Important Legal Notice
# MAGIC Bertelsmann Sandbox users get the right to work with the RTL2 Fernsehen GmbH & Co. KG datasets described below for the sole purpose of solving data mining and analytics tasks within the sandbox environment. The datasets may not be used in any other application context. 
# MAGIC 
# MAGIC Specifically, the use of the RTL2 Fernsehen GmbH & Co. KG datasets is subject to the following restrictions: 
# MAGIC 1. Following any RTL2 Fernsehen GmbH & Co. KG dataset download by a sandbox user, the user will delete this dataset from his/her systems within two weeks after the completion of the corresponding data mining/analytics task.
# MAGIC 2. By accessing any RTL2 Fernsehen GmbH & Co. KG dataset provided within the sandbox, the sandbox user agrees to restriction (1). <br><br>

# COMMAND ----------

# MAGIC %md # Sentiment Analysis of Social Media Comments
# MAGIC Nowadays, we can hardly imagine everyday life without social media. We use it to communicate with each other, share content and our opinions. For companies, social listening has become a way to get to know about consumer's opinions; track and analyse conversations, and ultimately, derive actions from it. **Sentiment Analysis** is a subset of social listening that focuses on retrieving information about a consumer's perception of a product, brand or service. <br> <br>
# MAGIC **RTLZWEI** is a private television broadcaster which is interested to learn more about the audience's opinion and feelings towards their daily soap __Köln 50667__. Therefore, they are seeking to implement an algorithm that helps them to understand the emotions and opinions of viewers expressed in **comments on their Köln 50667 Facebook page**. <br><br>

# COMMAND ----------

# MAGIC %md ## Background
# MAGIC 
# MAGIC 
# MAGIC **RTLZWEI** is a German-language television channel, which is operated by RTL2 Television GmbH & Co. KG. With Bertelsmann as one of three independent RTLZWEI shareholders the company is present across all relevant digital channels and its soaps produce ever increasing reach across the digital universe.
# MAGIC <br>
# MAGIC **RTLZWEI's** Data Science team is part of the Insights & Data Analytics department which supports both the editorial and the marketer side of our digital media house - across all channels. Having a deep understanding of interests and sentiments of our fans is crucial to support both strategic and operative decision making.
# MAGIC 
# MAGIC <img src="https://cm230.github.io/RTLZwei.pjpg" width="60%" height="60%"  align="center"/>
# MAGIC 
# MAGIC <br>
# MAGIC <br>
# MAGIC 
# MAGIC #### Daily Soap 'Köln 50667' by RTLZWEI
# MAGIC RTLZWEI is the Nr. 1 German-language reality TV channel, offering a program that shows life as it is: optimistic, surprising, and realistic. The program covers documentaries, scripted-reality shows, and high-quality movies, amongst others.
# MAGIC <br>
# MAGIC RTLZWEI's daily soap __Köln 50667__ is produced as a scripted-reality show. It is broadcasted in the afternoon from Monday to Friday. As the title of the soap __Köln 50667__ suggests, the plot takes place in Cologne, with **50667** being the postal code of the center of Cologne. The plot takes place on the streets of Cologne and at the bar of Alex. The soap shows the daily life of Alex and his crew. Stories are told about hope, love, success and drama.
# MAGIC <br>
# MAGIC During the past months the official __Köln 50667__ Facebook page produced an average monthly reach of more than 28 Mio.
# MAGIC <br>
# MAGIC __Köln 50667__ is targeted at young people. The daily soap is available on different social media channels, which gives fans the opportunity to dive into their idols' lives and keep viewers in touch with the show. __Köln 50667__ has been successful on social media from the beginning and is one of the most successful daily soaps by **RTLZWEI**.
# MAGIC <br>
# MAGIC __Köln 50667__ has its own Facebook channel with about 1.4 mio. subscribers and about the same number of likes. **RTLZWEI** creates posts on Facebook from the perspective of a character (In-universe posting) to give users the feeling of being able to interact with their idol. With the Facebook channel, **RTLZWEI** is seeking to increase the viewers' loyalty by bringing the soap to social media. 

# COMMAND ----------

# MAGIC %md ## Datasets
# MAGIC 
# MAGIC A. __Dataset with all comments__: __"FB_comments_sandbox_challenge.xlsx"__  <br>A dataset with approximately 84k user comments (~rows) from Köln 50667's own Facebook channel, from 1st June 2019 - 31st October 2019. <br><br>
# MAGIC __Variables__:
# MAGIC * __messageId__: unique ID of a user comment
# MAGIC * __postId__: unique ID of the Facebook post the comment refers to
# MAGIC * __PostMessage__: original text of the Facebook post
# MAGIC * __createdTime__: date and time when comment was created
# MAGIC * __message__: original text of user comment (most likely in German as the daily soap as well as the Facebook postings are in German)
# MAGIC * __GoogleTranslated__: original text (column "message") was translated into English using Google Translate
# MAGIC 
# MAGIC B. __Train dataset__: A dataset that contains the data from __dataset A: "FB_comments_sandbox_challenge.xlsx"__ excluding the Facebook comments used for validation and testing purposes. 
# MAGIC 
# MAGIC C. __Validation dataset__: __'FB_comments_validation_data.xlsx'__ <br> A dataset with approximately 1k user comments out of the 84k comments, which are manually labelled as positive, negative or neutral. This dataset can be used for the validation of your solution. <br><br>
# MAGIC * __messageId__: see above
# MAGIC * __createdTime__: see above
# MAGIC * __message__: see above
# MAGIC * __positive__: user comment was labelled as positive if the user comment conveyed a positive sentiment
# MAGIC * __negative__: user comment was labelled as negative if the user comment conveyed a negative sentiment
# MAGIC * __uncertain__: user comment was labelled as uncertain if sentiment conveyed by the user comment was not clear <br><br>
# MAGIC For the evaluation of the final solution, we have kept aside an additional __test dataset of 1k user comments__ which are manually labelled as well. The test dataset will not be provided.
# MAGIC For more information, please see description "Evaluation Performance VaderSentiment".

# COMMAND ----------

# MAGIC %md # Problem Statement
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC **RTLZWEI** is interested to learn more about the audience's opinion and feelings towards their daily soap __Köln 50667__. Therefore, they are seeking to implement an algorithm that helps them to understand the emotions and opinions of viewers expressed in **comments on their Köln 50667 Facebook page**. <br>
# MAGIC 
# MAGIC The task of this challenge is to develop an algorithm which classifies the comments as positive, negative or unclear using a lexicon-based sentiment analysis tool, deep learning or other methods. 
# MAGIC 
# MAGIC 
# MAGIC ## Evaluation Criteria
# MAGIC 
# MAGIC The sentiment classification (positive/negative/neutral) is evaluated based on the accuracy score on the test set according to the following framework:
# MAGIC  
# MAGIC $$\begin{align*}
# MAGIC & 0.5 < \text{score} \leq 0.7 \text{: 5 points}\\
# MAGIC & 0.7 < \text{score} \leq 0.85 \text{: 10 points}\\
# MAGIC & 0.85 < \text{score} \text{: 15 points}
# MAGIC \end{align*}$$
# MAGIC  
# MAGIC The team with the highest accuracy score receives an additional __5__ points.
# MAGIC  
# MAGIC ### Bonus Points
# MAGIC * **Emotion Detection**: Additional points can be collected if the approach additionally enables the detection of emotions such as angry, happy, amused etc. Submitted solutions are evaluated in terms of creativity and practicability. A maximum of __15__ bonus points can be achieved (partial points possible).
# MAGIC * **German Comments**: A solution that uses the German language or includes an automated translation process and has an accuracy over 0.85 on the test data receives __10__ bonus points.
# MAGIC * **API interface**: __5__ bonus points
# MAGIC * **Creativity**: We may award a further __5__ bonus points for outstanding visualizations or other business-relevant insights.
# MAGIC 
# MAGIC ##### The team with the highest number of points will be the winning team. 
# MAGIC <br><br><br>
# MAGIC  
# MAGIC  
# MAGIC  

# COMMAND ----------

# MAGIC %md 
# MAGIC # Showcase: Sentiment Analysis using VaderSentiment
# MAGIC 
# MAGIC In the following, we demonstrate an approach for the first difficulty level that should help you to get an idea for your approach.
# MAGIC <br> __VADER__ (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, but also works with other text types.

# COMMAND ----------

# MAGIC %md ## Import Libraries

# COMMAND ----------

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import VaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pathlib import Path

import time

analyzer = SentimentIntensityAnalyzer()


from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from PIL import Image

import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text.freqdist import FreqDistVisualizer
from yellowbrick.style import set_palette

import nltk
from nltk import tokenize

nltk.download("punkt")
# download set of stop words
nltk.download("stopwords")

import glob
import os

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.metrics import accuracy_score

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

from IPython.display import Image

# COMMAND ----------

# MAGIC %md ## Import Data

# COMMAND ----------

# current working directory
path = os.getcwd()

# COMMAND ----------

#read excel file
df = pd.read_excel((os.path.join(path, r'FB_comments_sandbox_challenge.xlsx')), index=False)
df.head()

# COMMAND ----------

#check how many rows and columns
df.shape

# COMMAND ----------

#make copy
data = df.copy()

# COMMAND ----------

# MAGIC %md ## Apply Vader

# COMMAND ----------

# obtain the polarity indices for the given sentence
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


# obtain only the negative score
def negative_score(text):
    negative_value = analyzer.polarity_scores(text)["neg"]
    return negative_value


# obtain only the neutral score
def neutral_score(text):
    neutral_value = analyzer.polarity_scores(text)["neu"]
    return neutral_value


# obtain only the positive score
def positive_score(text):
    positive_value = analyzer.polarity_scores(text)["pos"]
    return positive_value


# obtain only the compound score
def compound_score(text):
    """
    The Compound score is a metric that calculates the sum of all the lexicon ratings 
    which have been normalized between -1(most extreme negative) and +1 (most extreme positive)
    """
    compound_value = analyzer.polarity_scores(text)["compound"]
    return compound_value


# create new columns with sentiment scores
def create_sentiment_column(df):
    # drop rows where column GoogleTranslated is empty
    df.dropna(subset=["GoogleTranslated"], inplace=True)
    # create new columns
    df["pos_sentiment"] = df["GoogleTranslated"].apply(positive_score)
    df["neg_sentiment"] = df["GoogleTranslated"].apply(negative_score)
    df["neu_sentiment"] = df["GoogleTranslated"].apply(neutral_score)
    df["compound_sentiment"] = df["GoogleTranslated"].apply(compound_score)
    return df

# COMMAND ----------

# MAGIC %md VaderSentiment was applied on the dataset A: "FB_comments_sandbox_challenge.xlsx" because it is a lexicon-based approach that does not really train a model but rather applies the knowledge from the lexicon to the dataset.

# COMMAND ----------

# remove newlines \n and carriage returns \r
data = (
    data.replace({r"\s+$": "", r"^\s+": ""}, regex=True)
    .replace(r"\n", " ", regex=True)
    .replace(r"\r", " ", regex=True)
)

# COMMAND ----------

data = create_sentiment_column(data)
data.head()

# COMMAND ----------

# MAGIC %md ## Distribution of all scores

# COMMAND ----------

# plot histogram for each score
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
data.hist("pos_sentiment", bins=25, ax=axes[0, 0], color="mediumseagreen")
axes[0, 0].set_title("Positive Sentiment Score")
data.hist("neg_sentiment", bins=25, ax=axes[0, 1], color="r")
axes[0, 1].set_title("Negative Sentiment Score")
data.hist("neu_sentiment", bins=25, ax=axes[1, 0], color="slategrey")
axes[1, 0].set_title("Neutral Sentiment Score")
data.hist("compound_sentiment", bins=25, ax=axes[1, 1], color="darkcyan")
axes[1, 1].set_title("Compound")

# plot labels
fig.text(0.5, 0.04, "Sentiment Scores", fontweight="bold", ha="center")
fig.text(
    0.04, 0.5, "Number of Comments", fontweight="bold", va="center", rotation="vertical"
)

# plot title
plt.suptitle(
    "Sentiment Analysis of Facebook Comments Köln50667\n",
    fontsize=18,
    fontweight="bold",
)

# COMMAND ----------

# MAGIC %md - the first three plots represent the proportion of the comments that fall into each category
# MAGIC - compound score: sum of all of the lexicon ratings which have been standardized to range between -1 and 1, e.g. 0.2 would be rather neutral

# COMMAND ----------

# MAGIC %md ## Analyze tendency comments

# COMMAND ----------

percentiles = data.compound_sentiment.describe(
    percentiles=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
percentiles

# COMMAND ----------

# MAGIC %md #### Create separate dataframes for positive, negative, and neutral comments

# COMMAND ----------

#positive threshold for compound sentiment
pos_thresh = 0.1

#negative threshold
neg_thresh = -0.1

# COMMAND ----------

# dataframe with positive comments
df_pos = data.loc[data.compound_sentiment > pos_thresh]

# list with only positive comments
pos_comments = df_pos["message"].tolist()

# COMMAND ----------

# dataframe with negative comments
df_neg = data.loc[data.compound_sentiment < neg_thresh]

# list with only negative comments
neg_comments = df_neg["message"].tolist()

# COMMAND ----------

# dataframe with neutral comments
df_neutr = data[data['compound_sentiment'].between(neg_thresh, pos_thresh)]

# list with only neutral comments
neu_comments = df_neutr["message"].tolist()

# COMMAND ----------

# MAGIC %md ## Wordcloud

# COMMAND ----------

# plot wordcloud
def plot_wordcloud(wordcloud, text):
    plt.figure(figsize=(12, 11))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(text + " Comments\n", fontsize=18, fontweight="bold")
    plt.show()

# COMMAND ----------

# remove German stop words
stop_words = nltk.corpus.stopwords.words("german")
newStopWords = [
    "mal",
    "mehr",
    "immer",
    "und",
    "die",
    "da",
    "einfach",
    "geht",
    "ja",
    "hast",
    "schon",
    "sowas",
    "gibt",
    "wäre",
    "finde",
    "echt",
    "macht",
    "ne",
    "ganz",
    "beiden",
    "genau",
    "kommt",
    "sieht",
    "voll",
    "genau",
    "super",
    "besser",
    "tut",
    "wer",
    "nein",
    "leider",
    "gut",
    "zusammen",
    "oh"
]
stop_words.extend(newStopWords)

# COMMAND ----------

# Create dataframe only with positive text
text_df = df_pos[["message"]]
# join and put words in lower case
text = " ".join(words for words in text_df.message.str.lower())

# Generate a word cloud image
wordcloud = WordCloud(
    stopwords=stop_words, background_color="white", collocations=False
).generate(text)

plot_wordcloud(wordcloud, "\nPositive")

# COMMAND ----------

# Create dataframe only with negative comments
text_df_neg = df_neg[["message"]]
# join and put words in lower case
text_neg = " ".join(words for words in text_df_neg.message.str.lower())

# Generate a word cloud image
wordcloud = WordCloud(
    stopwords=stop_words, background_color="white", collocations=False
).generate(text_neg)

plot_wordcloud(wordcloud, "\nNegative")

# COMMAND ----------

# MAGIC %md ## Frequency positive and negative words

# COMMAND ----------

# plot frequency positive or negative words
def frequency_words(pos_comments, title):
    # vectorize text
    vectorizer = CountVectorizer(stop_words=stop_words)
    docs = vectorizer.fit_transform(pos_comments)
    features = vectorizer.get_feature_names()

    # prepare the plot
    set_palette("Set1")
    plt.figure(figsize=(15, 8))
    plt.title(title, fontsize=15)

    # plot top 30 most frequent terms
    visualizer = FreqDistVisualizer(features=features, n=30)
    visualizer.fit(docs)
    visualizer.poof

# COMMAND ----------

# plot frequency positive words
frequency_words(pos_comments, title="Top 30 positive comments")

# COMMAND ----------

# plot frequency negative words
frequency_words(neg_comments, title="Top 30 negative comments")

# COMMAND ----------

# check where a certain term appears in the negative comments
df_neg[df_neg["message"].str.contains("Serie")]

# COMMAND ----------

# print only the comments with that specific word
def check_comments(sentence, words):
    res = [all([k in s for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]

# COMMAND ----------

# check first ten negative comments where that specific word appears
words = ["Serie"]
print(check_comments(neg_comments, words)[:10])

# COMMAND ----------

# MAGIC %md ## Count Sentiment per Day

# COMMAND ----------

# set date as index
def df_set_index(df_pos):
    # copy df positive or negative comments
    df = df_pos.copy()
    # change to date time
    df["createdTime"] = pd.to_datetime(df["createdTime"])
    # set date as index
    df.set_index(["createdTime"], inplace=True)
    return df

# COMMAND ----------

# set date as index for df with positive comments
df_pos_day = df_set_index(df_pos)

# set date as index for df with negative comments
df_neg_day = df_set_index(df_neg)

# set date as index for df with neutral comments
df_neutr_day = df_set_index(df_neutr)

# COMMAND ----------

# create new df with count of positive and negative comments per day

# create empty dataframe
df_daily = pd.DataFrame()

# create column with positive count
df_daily["pos_count"] = df_pos_day.messageId.resample("D").count()

# create column with negative count
df_daily["neg_count"] = df_neg_day.messageId.resample("D").count()

# create column with neutral count
df_daily["neutr_count"] = df_neutr_day.messageId.resample("D").count()

# COMMAND ----------

# Create the plot space upon which to plot the data
fig, ax = plt.subplots(figsize=(15, 8))

# Add the x-axis and the y-axis to the plot
ax.plot(df_daily.index.values, df_daily["pos_count"], df_daily["neg_count"], "r")

# Set title and labels for axes
fig.suptitle("Sentiment Facebook Comments\nKöln 50667", fontsize=18)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Count comments", fontsize=14)
plt.xticks(rotation=70)
plt.gca().legend(("pos_count", "neg_count"), fontsize=14)

# set different frequency xticks, one tick for each week
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
# format date month and day
ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))

plt.show()

# COMMAND ----------

# MAGIC %md ## Share of Sentiment per Day

# COMMAND ----------

# create column with percentage count for each sentiment

def percentage_column (df, column_title = "positive %", sent_count = 'pos_count' ):
    df[column_title] = df[sent_count]/(df['pos_count'] + df['neg_count'] + df['neutr_count'])
    return df

# COMMAND ----------

#create percentage positive, negative and neutral sentiment
pos_col = percentage_column(df_daily, "positive %", "pos_count" )

neg_col = percentage_column(df_daily, "negative %", "neg_count" )

neutr_col = percentage_column(df_daily, "neutral %", "neutr_count" )

df_daily.head()

# COMMAND ----------

# Create the plot space upon which to plot the data
fig, ax = plt.subplots(figsize=(15, 8))

# Add the x-axis and the y-axis to the plot
ax.plot(df_daily.index.values, df_daily["positive %"], df_daily["negative %"], "r", df_daily["neutral %"])

# Set title and labels for axes
fig.suptitle("Sentiment Facebook Comments Köln 50667 \n Share per Day", fontsize=18)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Share", fontsize=14)
plt.xticks(rotation=70)
plt.gca().legend(("positive %", "negative %", "neutral %" ), fontsize=14)

# set different frequency xticks, one tick for each week
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
# format date month and day
ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))

plt.show()

# COMMAND ----------

# MAGIC %md ## Evaluation Performance VaderSentiment
# MAGIC For measuring how well VaderSentiment classified the polarity of the Facebook comments, we provide a labelled validation dataset. 4000 out of the >80k comments were manually labelled as positive, negative or uncertain.
# MAGIC <br>
# MAGIC 
# MAGIC Especially when dealing with social media text, it is often not very clear if a text conveys a positive or rather negative sentiment. Therefore, we have decided on the following rules to label the Facebook comments: 
# MAGIC <br><br>
# MAGIC <i><b>Positive, e.g. the author of that comment sounds as if:</b></i> 
# MAGIC 
# MAGIC <ul>
# MAGIC     <li> he/she feels entertained </li>
# MAGIC     <li> ... is having fun </li>
# MAGIC     <li> ... is maybe disagreeing with a certain fact/ situation, but still expressing him-/herself cheerfully about that </li>
# MAGIC     <li> ... shows compassion towards a certain fact </li>
# MAGIC     
# MAGIC </ul>
# MAGIC 
# MAGIC <i><b>Negative, e.g. the author of that comment sounds as if:</b></i> 
# MAGIC 
# MAGIC <ul>
# MAGIC     <li> he/she is angry </li>
# MAGIC     <li> ... sad </li>
# MAGIC     <li> ... annoyed  </li>
# MAGIC     
# MAGIC </ul>
# MAGIC 
# MAGIC <i><b>Neutral/uncertain, e.g. if:</b></i> 
# MAGIC 
# MAGIC <ul>
# MAGIC     <li> the sentiment is not clear </li>
# MAGIC     <li> much interpretation is necessary </li>
# MAGIC     
# MAGIC </ul>
# MAGIC 
# MAGIC A cross check of the labels was performed, keeping only the comments that were labelled identically by two people independently from each other so that we are more "confident" about the labels. Also, not relevant comments were removed, e.g. linked people, links to websites, etc.
# MAGIC 
# MAGIC In total, 2244 comments were left. For the validation dataset, we provide you 1009 labelled comments. The rest is kept as test dataset.

# COMMAND ----------

# MAGIC %md ## Caution
# MAGIC __The meaning of the neutral label is ambiguous__ and should be interpreted with caution. As mentioned, we manually labelled those comments in the test and validation dataset as neutral/uncertain where the sentiment was not clear, appeared both positive as well as negative to us, or much interpretation was necessary. However, when VaderSentiment classifies a comment as neutral, it means that VaderSentiment was not able to classify the comments at all. <br>Please keep this in mind when interpreting the results.

# COMMAND ----------

#read excel file validation data
df_val = pd.read_excel((os.path.join(path, r'FB_comments_validation_data.xlsx')), index=False)
df_val.head()

# COMMAND ----------

# MAGIC %md ## Evaluation

# COMMAND ----------

# MAGIC %md #### Create a new column "y_val" that captures all sentiment labels in one
# MAGIC The labels are assigned as follows: <br>
# MAGIC __positive__: y=1 <br>
# MAGIC __negative__: y=0 <br>
# MAGIC __neutral/uncertain__: y=2 <br>

# COMMAND ----------

"""
assign labels:
1 (positive)
0 (negative)
2 (neutral) 

"""

#function to create dataframe with new column that captures all labels: positive, negative, and neutral

def replace_values(df, column="y_val"):
    #create a copy of the original df
    df_new = df.copy()
    
    #copy the labels of the positive column as positive is already positive=1 and negative=0 there
    df_new["y_val"] = df_new["positive"]
    
    #get index of rows that were labelled as neutral: not clearly positive or negative
    index_uncertain = df_new[(df_new['positive'] ==0) & (df_new['negative'] ==0) & (df_new['uncertain'] ==1)].index
    index_pos_n_neg = df_new[(df_new['positive'] ==1) & (df_new['negative'] ==1) & (df_new['uncertain'] ==0)].index
    
    #Replace values of neutral comments in the column "y_test" with 2 
    df_new.loc[index_uncertain, column] = 2
    df_new.loc[index_pos_n_neg, column] = 2
    return df_new


# COMMAND ----------

# create dataframe with new column "y_val"
df_val_new = replace_values(df_val, column="y_val")
df_val_new.head()

# COMMAND ----------

# dataframe containing the column with the manually assigned label and the calculated sentiment by Vader
df_val_merged = pd.merge(
    df_val_new[["messageId", "createdTime", "message", "y_val"]],
    data[["messageId", "compound_sentiment"]],
    on="messageId",
    how="left",
)
df_val_merged.head()

# COMMAND ----------

# check how many comments were labelled as neutral according to Vader

#number
print(
    "Number of comments Vader labelled as neutral: {}".format(
        df_val_merged["compound_sentiment"].isin([0]).sum()
    )
)

#Percentage
print(
    "Percentage neutral: {:.2%}".format(
        (df_val_merged["compound_sentiment"].isin([0]).sum()/len(df_val_merged))
    )
)


# COMMAND ----------

# MAGIC %md #### Create a new column in the dataframe that maps label to the compound score calculated by Vader
# MAGIC 
# MAGIC The labels are assigned as follows: <br>
# MAGIC __positive__: y=1 if compound score >0 <br>
# MAGIC __negative__: y=0 if compound score <0 <br>
# MAGIC __neutral/uncertain__: y=2 y=0 if compound score =0 <br>

# COMMAND ----------

"""
assign labels:
1 (positive) if larger than 0
0 (negative) if smaller than 0
2 (neutral) if else

"""
df_val_merged["y_pred"] = df_val_merged.compound_sentiment.map(
    lambda x: int(1) if x > 0 else int(0) if x < 0 else int(2)
)

df_val_merged.head()

# COMMAND ----------

# accuracy score
print(
    "Accuracy Score: {:.2%}".format(
        accuracy_score(df_val_merged.y_val, df_val_merged.y_pred)
    )
)

# COMMAND ----------

# Create a confusion matrix


def plot_cm(y_val, y_pred, part_title, classes=[0, 1, 2]):
    # Creates a confusion matrix
    classes = classes
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    # Plot Confusion Matrix
    f, ax = plt.subplots(figsize=(9,7 ))

    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=True)
    plt.title(part_title.format(accuracy_score(y_val, y_pred)))
    plt.ylabel("True label (manually labelled)")
    plt.xlabel("Predicted label (Vader)")
    plt.show()

# COMMAND ----------

# Create a confusion matrix
plot_cm(
    df_val_merged.y_val,
    df_val_merged.y_pred,
    part_title="Confusion Matrix \nNegative=0, Positive=1,  Neutral=2 \nAccuracy:{0:.3f}",
    classes=[0, 1, 2],
)