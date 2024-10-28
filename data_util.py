
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


def append_sentiment_to_tweet():
    