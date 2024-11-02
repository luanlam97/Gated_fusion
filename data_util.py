
import torch
import kagglehub
import pandas as pd
from torch.utils.data import  DataLoader
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
from torch.nn.functional import softmax
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data_from_kaggle():
    sentiment_data_path = kagglehub.dataset_download("equinxx/stock-tweets-for-sentiment-analysis-and-prediction")
    stock = pd.read_csv(sentiment_data_path + '/stock_yfinance_data.csv')
    tweet = pd.read_csv(sentiment_data_path + '/stock_tweets.csv')
    return stock, tweet

def sentiment_analysis(tweet_df, device = 'cuda'):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment = {   'Positive' : [],
                    'Neutral' : [], 
                    'Negative' : []} 
    for i in tweet_df['Tweet']:
        input = tokenizer(i, return_tensors="pt")
        input.to(device)
        output = model(**input)
        scores = softmax(output[0][0])
        scores = scores.cpu().detach().numpy()
        sentiment['Positive'].append(scores[0])
        sentiment['Neutral'].append(scores[1])
        sentiment['Negative'].append(scores[2])
    tweet_df = tweet_df.assign(**sentiment)
    return tweet_df

def aggregate_tweet(tweet_df):
    tweet_df['Date'] = pd.to_datetime(tweet_df['Date']).dt.date
    tweet_df['overal sentiment'] = tweet_df[['Positive', 'Neutral', 'Negative']].idxmax(axis=1)
    aggregated_tweet = tweet_df.groupby(['Stock Name', 'Date']).agg(
        Positive_Avg=('Positive', 'mean'),
        Neutral_Avg=('Neutral', 'mean'),
        Negative_Avg=('Negative', 'mean'),
        Positive_Count=('overal sentiment', lambda x: (x == 'Positive').sum()),
        Neutral_Count=('overal sentiment', lambda x: (x == 'Neutral').sum()),
        Negative_Count=('overal sentiment', lambda x: (x == 'Negative').sum())
    ).reset_index()
    return aggregated_tweet

def restructure_date_information(df):
    df["Month"] = pd.to_datetime(df['Date']).dt.month 
    df["Day"] = pd.to_datetime(df['Date']).dt.day
    df["Day of The Week"] = pd.to_datetime(df['Date']).dt.dayofweek
    df["Week of The Year"] = pd.to_datetime(df['Date']).dt.isocalendar().week
    df.drop('Date', axis=1, inplace=True)
    return df