import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset
#from transformers import AutoTokenizer,  AutoModelForSequenceClassification

class TFT_Dataset(Dataset):
    def __init__(self , stock_df, static_df , constant_variable, history_length = 90, prediction_length = 5, device = 'cuda'):
        self.device = device
        self.hist_cat_feature = [feature for feature, type in constant_variable.feature_variables.items() if type =='Categorical']
        self.hist_cont_feature = [feature for feature, type in constant_variable.feature_variables.items() if type =='Numerical']
        
        self.static_cat_feature =  [feature for feature, type in constant_variable.static_variables.items() if type =='Categorical']
        self.static_cont_feature =  [feature for feature, type in constant_variable.static_variables.items() if type =='Numerical']
        
        self.constant_variable = constant_variable
        self.data_cont = stock_df.groupby('Stock Name').apply(lambda x: x[self.hist_cont_feature].values.tolist() , include_groups=False).to_dict()
        self.data_cat = stock_df.groupby('Stock Name').apply(lambda x: x[self.hist_cat_feature].values.tolist() , include_groups=False).to_dict()

        self.future_data = stock_df.groupby('Stock Name') \
                                .apply(lambda x: x[self.constant_variable.future_feature.keys()]\
                                .values.tolist() , include_groups=False).to_dict()
        self.prediction = stock_df.groupby('Stock Name').apply(lambda x: x[constant_variable.prediction_feature.keys()].values.tolist() , include_groups=False).to_dict()
        
        self.static_df = static_df
        self.history_length = history_length
        self.prediction_length = prediction_length
        
        self.data_index = []
        for name in self.data_cont.keys():
            for idx in range(252- self.history_length - self.prediction_length ):
                self.data_index.append( (name, idx)) 

        

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        stock_name , idx  = self.data_index[idx]  
        static_cont_input = self.static_df[self.static_cont_feature].loc[stock_name]
        static_cat_input = self.static_df[self.static_cat_feature].loc[stock_name]
        

        history_cont_input = self.data_cont[stock_name][idx: idx + self.history_length]
        history_cat_input = self.data_cat[stock_name][idx: idx + self.history_length]
        future_input = self.future_data[stock_name][idx + self.history_length: idx + self.history_length + self.prediction_length]
        prediction = self.prediction[stock_name][idx+ self.history_length : idx + self.history_length + self.prediction_length]

        static_cont_input = torch.tensor(static_cont_input.values, device= self.device).float()
 
        static_cat_input = torch.tensor(static_cat_input.values, device= self.device)
        
        history_cont_input = torch.tensor(history_cont_input, device= self.device).float()
        history_cat_input = torch.tensor(history_cat_input, device= self.device)

        future_input = torch.tensor(future_input , device= self.device)
        prediction = torch.tensor(prediction, device= self.device)


      
        return static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input, prediction