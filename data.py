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

        # will plot open prices throughout each day, use it for prediction
        open_prices = np.array(history_cont_input)[:, 0]   # y-axis: open prices
        days = np.arange(self.history_length)  # x-axis: day indices (0, 1, ..., history_length-1)

        # plot the image
        plt.switch_backend('Agg') # make ssure it doesn't print on my output to clog it
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(days, open_prices)
        ax.axis('off')  # Hide the axis

        # https://stackoverflow.com/questions/67955433/how-to-get-matplotlib-plot-data-as-numpy-array

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # H x W x C format

        pil_image = Image.fromarray(image)
        img = transforms.Compose([
                        transforms.Resize((224 , 224 )), 
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5),
                    ])(pil_image).to(self.device)

        # # patching
        # grid_size = 4  # 4x4 grid
        # image_size = 64 # (arbitrary for now)
        # image_resized = np.resize(image_gray, (image_size, image_size)) # resize image
        # patch_size = image_size // grid_size  # size of each patch (16x16)

        # # split image into 4x4 grid of 16x16 patches
        # patches = []
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         patch = image_resized[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
        #         patches.append(patch)

        # # convert patches to tensor
        # # use stack to turn into tenssor instead of 
        # patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32)  # shape: (16, 16, 16)

        # # check tensor shape (should be (16, 16, 16) for 16 patches each 16x16
        # # print("Tensor Shape:", patches_tensor.shape)
        # plt.close(fig)  # for memory


        # get the adjusted close price at day 90
        price_day_90 = self.prediction[stock_name][idx+ self.history_length]

        # get the average adjusted close price at 15 days after 90
        prices_day_90_to_105 = self.prediction[stock_name][idx+ self.history_length :idx+ self.history_length + self.prediction_length ]
        avg_price_day_90_to_105 = np.mean(prices_day_90_to_105)


        # calculate the target: 1 if price goes up, 0 if it goes down
        # at the end of that 15 days
        prediction_classification = [1,0] if avg_price_day_90_to_105 > price_day_90 else [0,1]
        prediction_classification = torch.tensor(prediction_classification, device=self.device).float() # add extra dimension at -1
        # squeeze deletes dimension

        ####################################################################################

        static_cont_input = torch.tensor(static_cont_input.values, device= self.device).float()
        static_cat_input = torch.tensor(static_cat_input.values, device= self.device)
        
        history_cont_input = torch.tensor(history_cont_input, device= self.device).float()
        history_cat_input = torch.tensor(history_cat_input, device= self.device)

        future_input = torch.tensor(future_input , device= self.device)
        prediction = torch.tensor(prediction, device= self.device)
        img.to
        return static_cont_input, static_cat_input,history_cont_input, history_cat_input,future_input, prediction, img, prediction_classification




