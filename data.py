import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,  AutoModelForSequenceClassification

class TFT_Dataset(Dataset):
    def __init__(self , df,  history_length = 90, prediction_length = 5, device = 'cuda'):
        self.data = df.groupby('Stock Name').apply(lambda x: x.values.tolist() , include_groups=False).to_dict()
        self.future_data = df.groupby('Stock Name').apply(lambda x: x.iloc[:, -4:].values.tolist() , include_groups=False).to_dict()
        self.prediction = df.groupby('Stock Name')["Close"].apply(lambda x: x.values.tolist() , include_groups=False).to_dict()

        self.history_length = history_length
        self.prediction_length = prediction_length
        
        self.data_index = []
        for name in self.data.keys():
            for idx in range(252- self.history_length - self.prediction_length ):
                self.data_index.append( (name, idx)) 

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        stock_name , idx  = self.data_index[idx]  

        static_input = self.get_static(stock_name)
        history_input = self.data[stock_name][idx: idx + self.history_length ]
        future_input = self.future_data[stock_name][idx + self.history_length: idx + self.history_length + self.prediction_length]
        prediction = self.prediction[stock_name][idx+ self.history_length : idx + self.history_length + self.prediction_length]

        history_input = torch.tensor(history_input)
        future_input = torch.tensor(future_input)
        prediction = torch.tensor(prediction)

        return static_input, history_input, future_input, prediction

    def get_static(self, stock_name):
        return stock_name
