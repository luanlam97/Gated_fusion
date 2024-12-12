import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import cupy as cp
from PIL import Image
from torchvision import transforms

class ViT_Dataset(Dataset):
    def __init__(self , stock_df, 
                        static_df , 
                        constant_variable, 
                        history_length = 90, 
                        prediction_length = 5, 
                        device = 'cuda'):
                
        self.stock_df = stock_df
        self.static_df = static_df
        self.constant_variable = constant_variable
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.device = device

        ##########
        # for creating images
        # saving images to a directory if not created
        # i think the runtime issue was graphing the images each time
        self.image_dir = '/content/images'  
        os.makedirs(self.image_dir, exist_ok=True)      

        ###########

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

        history_cont_input_df = pd.DataFrame(history_cont_input, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

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

        ######
        # add line to get image
        ######

        ##################################################
        # for creating images
        # image_filename = os.path.join(self.image_dir, f"{stock_name}.png")
        image_files = os.listdir(self.image_dir)
        image_filename = f"{stock_name}_{idx}.png"
        
        image_path = os.path.join(self.image_dir, f"{stock_name}_{idx}.png")

        if image_filename in image_files:
          pil_image = Image.open(image_path)  # Open the existing image
          image_tensor = transforms.Compose([
                          transforms.Resize((64 , 64 )), 
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5),
                      ])(pil_image).to(self.device)
        else:
          # plots for image enrichment
          # moving averages (MA20, MA50) 
          ma20 = history_cont_input_df['Close'].rolling(window=20).mean()
          ma50 = history_cont_input_df['Close'].rolling(window=50).mean()

          # bollinger bands (BB) based on the MA20 and 20-day rolling std deviation
          bb_upper = ma20 + 2 * history_cont_input_df['Close'].rolling(window=20).std()
          bb_lower = ma20 - 2 * history_cont_input_df['Close'].rolling(window=20).std()

          # RSI (relative strength index, 14-day period)
          delta = history_cont_input_df['Close'].diff()
          gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
          loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
          rs = gain / loss
          rsi = 100 - (100 / (1 + rs))

          # Moving Average of Volume (VMA20) 
          vma20 = history_cont_input_df['Volume'].rolling(window=20).mean()

          # chaikin money flow (CMF)  based on the rolling window for volume and price
          cmf = ((history_cont_input_df['Close'] - history_cont_input_df['Low'] - (history_cont_input_df['High'] - \
          history_cont_input_df['Low']) / 2) * history_cont_input_df['Volume']) / (history_cont_input_df['High'] - \
          history_cont_input_df['Low'])

          # volume
          volume = history_cont_input_df['Volume']  # volume for the history window
          cmf = cmf[-self.history_length:]  # align cmf with history window
          ma20 = ma20[-self.history_length:]  # last `history_length` days of MA20 and MA50
          ma50 = ma50[-self.history_length:] 

          # history length days for bb upper and lower bands
          bb_upper = bb_upper[-self.history_length:] 
          bb_lower = bb_lower[-self.history_length:] 

          #`history_length` days of CMF
          cmf = cmf[-self.history_length:]  


          # will plot open prices throughout each day, use it for prediction
          open_prices = np.array(history_cont_input)[:, 0]   # y-axis: open prices
          days = np.arange(self.history_length)  # x-axis: day indices (0, 1, ..., history_length-1)

          # plot the image
          plt.switch_backend('Agg') # make ssure it doesn't print on my output to clog it
          fig, ax = plt.subplots(figsize=(10, 6))
          # Get the y-axis limits (lower and upper bounds of the plot's y-axis)
          y_min, y_max = ax.get_ylim()

          # plot the price
          ax.plot(days, open_prices, color='blue', linewidth=1)
          # ax.plot(days, close_prices, color='green', linewidth=1)

          # plot Moving Averages (MA20 and MA50)
          ax.plot(days, ma20, color='orange', linewidth=1)
          ax.plot(days, ma50, color='red', linewidth=1)

          # # plot Bollinger Bands
          ax.fill_between(days, bb_upper, bb_lower, color='gray', alpha=0.2)

          # # plot the CMF (Chaikin Money Flow)
          # ax.plot(days, cmf, color='brown', linestyle='--', linewidth=1)

          # # plot volume as a bar chart (twin axis)
          volume_factor = 0.5  # scale volume correctly
          max_volume_height = y_max
          # Ensure that the volume doesn't exceed the maximum height
          restricted_volume = np.clip(volume * volume_factor, 0, max_volume_height)
          ax.bar(days, volume * volume_factor, width=0.8, color='lightblue', alpha=0.5, label='Volume')

          # ax.plot(days, open_prices)
          ax.axis('off')  # Hide the axis

          fig.canvas.draw()
          image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
          image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # H x W x C format

          pil_image = Image.fromarray(image)
          image_tensor = transforms.Compose([
                          transforms.Resize((64 , 64 )), 
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5),
                      ])(pil_image).to(self.device)


          # Save the image to the /images directory
          image_filename = os.path.join(self.image_dir, f"{stock_name}_{idx}.png")
          pil_image.save(image_filename)  # Save image as PNG

          plt.close(fig)  # for memory      
        ##################################################


        # get the adjusted close price at avg day 0 to 90
        price_day_90 = self.prediction[stock_name][idx: idx+ self.history_length]
        avg_price_day_90 = np.mean(price_day_90)

        # get the average adjusted close price at 15 days after 90
        prices_day_90_to_105 = self.prediction[stock_name][idx+ self.history_length :idx+ self.history_length + self.prediction_length ]
        avg_price_day_90_to_105 = np.mean(prices_day_90_to_105)


        # calculate the target: 1 if price goes up, 0 if it goes down
        # at the end of that 15 days
        prediction_classification = [1,0] if avg_price_day_90_to_105 > avg_price_day_90 else [0,1]
        prediction_classification = torch.tensor(prediction_classification, device=self.device).float() # add extra dimension at -1
        # squeeze deletes dimension

        ####################################################################################

        static_cont_input = torch.tensor(static_cont_input.values, device= self.device).float()
        static_cat_input = torch.tensor(static_cat_input.values, device= self.device)
        
        history_cont_input = torch.tensor(history_cont_input, device= self.device).float()
        history_cat_input = torch.tensor(history_cat_input, device= self.device)

        future_input = torch.tensor(future_input , device= self.device)
        prediction = torch.tensor(prediction, device= self.device)

        return static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input, prediction, image_tensor, prediction_classification




