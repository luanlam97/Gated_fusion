import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset

class TFT_Dataset(Dataset):
    def __init__(self , stock_df, 
                        static_df , 
                        constant_variable, 
                        history_length = 90, 
                        prediction_length = 5, 
                        device = 'cuda'):
                
        self.static_df = static_df
        self.constant_variable = constant_variable
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.device = device

        self.hist_cat_feature = [feature for feature, type in constant_variable.feature_variables.items() 
                                        if type =='Categorical']
        self.hist_cont_feature = [feature for feature, type in constant_variable.feature_variables.items() 
                                        if type =='Numerical']
        
        self.static_cat_feature =  [feature for feature, type in constant_variable.static_variables.items() 
                                        if type =='Categorical']
        self.static_cont_feature =  [feature for feature, type in constant_variable.static_variables.items() 
                                        if type =='Numerical']
        
        
        self.data_cont = stock_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.hist_cont_feature].values.tolist() , include_groups=False) \
                                    .to_dict()
        self.data_cat = stock_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.hist_cat_feature].values.tolist() , include_groups=False) \
                                    .to_dict()

        self.future_data = stock_df.groupby('Stock Name') \
                                    .apply(lambda x: x[self.constant_variable.future_feature.keys()]\
                                    .values.tolist() , include_groups=False).to_dict()
        self.prediction = stock_df.groupby('Stock Name') \
                                    .apply(lambda x: x[constant_variable.prediction_feature.keys()].values.tolist() , include_groups=False) \
                                    .to_dict()

        self.data_index = []
        for name in self.data_cont.keys():
            if len(self.data_cont[name])- self.history_length - self.prediction_length > 0:
                for idx in range(len(self.data_cont[name])- self.history_length - self.prediction_length ):
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

        ####################################################################################
        # image creation here
        # history_cont_input is the stock history continuous data aka: open close, volume etc 
        # history_cont_input is a list of history_length day
        # so since default history_length = 90, history_cont_input is [day_0, day_1,....    day_89]
        # each day_x is a list of the stock features: [open_value, high_value, low_value, volume_value]
        # all data is scaled min-max in my preprocessing, (we can change to any other scalar if need)
        # You can convert image to np array then to torch tensor to avoid saving the images. 
        #
        # prediction variable is the 'Adjusted Close' target prediction (We can just pick 'Close' if needed, just change the parameters in constant.py)
        #           
        # You can build the image here tbh since history_cont_input is already processed
        # You can use any of the self. variable           
        # I believe we have to use same history_length for all model, aka same window length
        # but models can have different prediction lengths.          
        #
        ####################################################################################

        static_cont_input = torch.tensor(static_cont_input.values, device= self.device).float()
        static_cat_input = torch.tensor(static_cat_input.values, device= self.device)
        
        history_cont_input = torch.tensor(history_cont_input, device= self.device).float()
        history_cat_input = torch.tensor(history_cat_input, device= self.device)

        future_input = torch.tensor(future_input , device= self.device)
        prediction = torch.tensor(prediction, device= self.device)

        return static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input, prediction