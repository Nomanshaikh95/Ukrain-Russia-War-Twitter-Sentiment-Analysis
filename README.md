# Ukrain-Russia-War-Twitter-Sentiment-Analysis

•	Project aims to perform sentiment analysis on Twitter data related to the Ukraine-Russia war.
•	Involves collecting relevant tweets using appropriate keywords.
•	Pre-processing the data to prepare it for analysis.
•	Using natural language processing techniques to classify the sentiment of each tweet as positive, negative, or neutral.
•	Results of the analysis can provide insights into public opinion and attitudes towards the ongoing conflict.

# Python code  

# Data coleection
!pip install snscrape
# Data collection from Twitter
import snscrape.modules.twitter as sntwitter
import pandas as pd 

query = "Ukrain Russia War"
tweets =[]
limit = 3200

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit :
        break
    else:
        tweets.append(tweet.content)

df=pd.DataFrame(tweets)
print(df)

# save data into csv file
df.to_csv("UkrainRussiaWar.csv")

# Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\Projects\ukrain russia\filename.csv")
print(data.head())

# column names 
print(data.columns)

# Required columns
data = data[["username", "tweet", "language"]]

# Missing values and there sum
data.isnull().sum()

# check how many tweets are available in which languages  
data["language"].value_counts()

# Removing links, punctuation, symbols etc
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)


# wordcloud of most frequent words 
text = " ".join(i for i in data.tweet)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Adding and checking sentiment score
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data = data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())

# frequent word with positive sentiment
positive =' '.join([i for i in data['tweet'][data['Positive'] > data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# frequent word with negative sentiment
negative =' '.join([i for i in data['tweet'][data['Negative'] > data["Positive"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Summary
There are a lot of tweets about the Ukraine and Russia war where people tend to update about the ground truths, what they feel about it, and who they are supporting. I used those tweets for the task of Twitter sentiment analysis on the Ukraine and Russia war.
