class Constant():
    def __init__(self):

        self.static_variables = {
                "industry" : "Categorical", 
                "sector" : "Categorical", 
                "country" : "Categorical",
                "beta" : "Numerical",
                "marketCap" : "Numerical",
                "bookValue" : "Numerical",
                "dividendRate" : "Numerical",
                "dividendYield" : "Numerical",
                "fiveYearAvgDividendYield" : "Numerical",
                "debtToEquity": "Numerical"
            }

        self.feature_variables = {
            "Open" : "Numerical",
            "High" : "Numerical",
            "Low" : "Numerical",
            "Volume" : "Numerical",
            "Month"	 : "Categorical",
            "Day" : "Categorical",
            "Day of The Week"  : "Categorical",
            "Week of The Year" : "Categorical"
        }
        self.indicators = {
            'key' : 'values'
        }

        self.future_feature ={
            "Month"	 : "Categorical",
            "Day" : "Categorical",
            "Day of The Week"  : "Categorical",
            "Week of The Year" : "Categorical"
            }
        
        self.prediction_feature= {
            "Adjusted Close" : "Numerical",
        }

        self.columns_to_scale = ['Open',	'High',	'Low',	'Close',	'Adjusted Close',	'Volume']

        self.autoformer_future_feature ={
            "Open" : "Numerical",
            "High" : "Numerical",
            "Low" : "Numerical",
            "Close": "Numerical",
            'Adjusted Close':"Numerical",

            }
        self.autoformer_model_config ={
            'input_dim' : 5,
            'd_model' : 128,
            'n_heads' : 8,
            'ff_dim' : 128,
            'num_layers' : 2,
            'kernel_size' : 5,
            'target_len' : 7,
            'dropout' : 0.01
        } 