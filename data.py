import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from PIL import Image
from torchvision import transforms

class TFT_Dataset(Dataset):
    def __init__(self , tft_df, 
                        static_df , 
                        autoformer_df,
                        constant_variable, 
                        history_length = 90, 
                        tft_prediction_length = 5, 
                        autoformer_prediction_length = 7,
                        device = 'cuda'):
        self.static_df = static_df
        self.constant_variable = constant_variable
        self.history_length = history_length
        self.tft_prediction_length = tft_prediction_length
        self.autoformer_prediction_length = autoformer_prediction_length
        self.device = device
        self.hist_cat_feature = [feature for feature, type in constant_variable.feature_variables.items() 
                                        if type =='Categorical']
        self.hist_cont_feature = [feature for feature, type in constant_variable.feature_variables.items() 
                                        if type =='Numerical']
        
        self.static_cat_feature =  [feature for feature, type in constant_variable.static_variables.items() 
                                        if type =='Categorical']
        self.static_cont_feature =  [feature for feature, type in constant_variable.static_variables.items() 
                                        if type =='Numerical']
        
        
        self.tft_data_cont = tft_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.hist_cont_feature].values.tolist() , include_groups=False) \
                                    .to_dict()
        self.tft_data_cat = tft_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.hist_cat_feature].values.tolist() , include_groups=False) \
                                    .to_dict()

        self.tft_future_data = tft_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.constant_variable.future_feature.keys()]\
                                    .values.tolist() , include_groups=False).to_dict()
        self.tft_prediction = tft_df.groupby('Stock Name') \
                                    .apply(lambda x: x[constant_variable.prediction_feature.keys()].values.tolist() , include_groups=False) \
                                    .to_dict()
        
        self.autoformer_feature = [feature for feature, type in constant_variable.autoformer_future_feature.items() 
                                        if type =='Numerical']

        self.autoformer_data_cont = autoformer_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.autoformer_feature].values.tolist() , include_groups=False) \
                                    .to_dict()

        self.autoformer_prediction = autoformer_df.groupby('Stock Name') \
                                    .apply(lambda x: x[constant_variable.prediction_feature.keys()].values.tolist() , include_groups=False) \
                                    .to_dict()

        self.data_index = []
        for name in self.tft_data_cont.keys():
            if len(self.tft_data_cont[name])- self.history_length - self.tft_prediction_length > 0:
                for idx in range(len(self.tft_data_cont[name])- self.history_length - self.tft_prediction_length ):
                    self.data_index.append( (name, idx)) 

    def __len__(self):

        return len(self.data_index)

    def __getitem__(self, idx):
        stock_name , idx  = self.data_index[idx]  

        # TFT Section
        static_cont_input = self.static_df[self.static_cont_feature].loc[stock_name]
        static_cat_input = self.static_df[self.static_cat_feature].loc[stock_name]
        
        history_cont_input = self.tft_data_cont[stock_name][idx: idx + self.history_length]
        history_cat_input = self.tft_data_cat[stock_name][idx: idx + self.history_length]

        future_input = self.tft_future_data[stock_name][idx + self.history_length: idx + self.history_length + self.tft_prediction_length]
        tft_prediction = self.tft_prediction[stock_name][idx+ self.history_length : idx + self.history_length + self.tft_prediction_length]

        static_cont_input = torch.tensor(static_cont_input.values, device= self.device).float()
        static_cat_input = torch.tensor(static_cat_input.values, device= self.device)
        
        history_cont_input = torch.tensor(history_cont_input, device= self.device).float()
        history_cat_input = torch.tensor(history_cat_input, device= self.device)

        future_input = torch.tensor(future_input , device= self.device)
        tft_prediction = torch.tensor(tft_prediction, device= self.device)


        autoformer_feature_input = self.autoformer_data_cont[stock_name][idx: idx + self.history_length]
        autoformer_prediction = self.autoformer_prediction[stock_name][idx+ self.history_length : idx + self.history_length + self.autoformer_prediction_length]
        autoformer_feature_input = torch.tensor(autoformer_feature_input, device= self.device).float()
        autoformer_prediction = torch.tensor(autoformer_prediction, device= self.device).float()

        return static_cont_input, static_cat_input,history_cont_input, history_cat_input,future_input, tft_prediction, autoformer_feature_input, autoformer_prediction




