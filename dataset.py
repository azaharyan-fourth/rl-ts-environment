import numpy as np
import pandas as pd
from datetime import datetime

class Dataset:
        
    def __init__(self, file_path, start_test_period):
        df = pd.read_csv(file_path,
                            parse_dates=['date'],
                            date_parser=lambda x: datetime.strptime(str(x), '%m/%d/%Y'))

        df = df.set_index('index')
        df = df.sort_index()
        self.series = df

        #self.split_train = round(0.8*df.size)
        self.split_dataset(start_test_period)

    def split_dataset(self, start_test_period):
        #train-test split dataset
        self.dataset_train = self.series[self.series['date'] < start_test_period]
        self.dataset_val = self.series[(self.series['date'] >= start_test_period)]

    

      
