
import torch
import kagglehub
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import  LabelEncoder, MinMaxScaler
from torch.utils.data import  DataLoader
#from transformers import AutoTokenizer,  AutoModelForSequenceClassification
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

    df["Month"] = df["Month"]- 1
    df["Day"] = pd.to_datetime(df['Date']).dt.day
    df["Day"] = df["Day"]- 1
    df["Day of The Week"] = pd.to_datetime(df['Date']).dt.dayofweek

    df["Week of The Year"] = pd.to_datetime(df['Date']).dt.isocalendar().week
    df["Week of The Year"] = df["Week of The Year"]- 1
    df.drop('Date', axis=1, inplace=True)
    return df

def get_static_df(stock_df, static_variables):
    stock_list = stock_df["Stock Name"].unique()
    static_info = {}
    for stock in stock_list:
        stock = yf.Ticker(stock)
        info = stock.info
        selected_info = {var: info.get(var, None) for var in static_variables.keys()}
        static_info[stock.ticker] = selected_info

    static_df = pd.DataFrame.from_dict(static_info).T
    static_df.fillna(0, inplace=True)

    return static_df

def scale_stock_data(stock_df, column):
    scalar = {}
    out_df = pd.DataFrame(columns=column)
    scaled_data = []

    for stock in stock_df["Stock Name"].unique():
        scaler = MinMaxScaler()
        stock_data = stock_df[stock_df['Stock Name'] == stock].copy()
        stock_data[column] = scaler.fit_transform(stock_data[column])
        scaled_data.append(stock_data)
        scalar[stock] = scaler
    out_df = pd.concat(scaled_data, ignore_index=True)
    out_df["Date"] = stock_df['Date']

    return out_df, scalar

def stock_quantiles(df, quantiles):
    result = df.groupby("Stock Name")["Adj Close"].quantile(quantiles).unstack()
    result.columns = [f"q_{int(q*100)}" for q in quantiles]
    return result

def one_label_scale_static_df(static_df, static_variables):
    cat_variable =  [variable  for variable, value  in static_variables.items() if value == 'Categorical']
    num_variable =  [variable  for variable, value  in static_variables.items() if value == 'Numerical']
    for variable in cat_variable:
        le = LabelEncoder()
        static_df[variable] = le.fit_transform(static_df[variable])

    scaler = MinMaxScaler()
    static_df[num_variable] = scaler.fit_transform(static_df[num_variable])

    return static_df


def get_feature_length(df, constant):
    cat_variable =  [variable  for variable, value  in constant.items() if value == 'Categorical']
    num_variable =  [variable  for variable, value  in constant.items() if value == 'Numerical']
    cat_variable_list = []
    for variable in cat_variable:
        cat_variable_list.append(len(df[variable].unique()))

    return cat_variable_list, len(num_variable)




