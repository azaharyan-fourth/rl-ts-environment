from datetime import datetime
import pandas as pd

class Dataset:
    def __init__(self,
                file_path: str,
                start_test_period: str):
        df = pd.read_csv(file_path,
                            parse_dates=['date'],
                            date_parser=lambda x: datetime.strptime(str(x), '%m/%d/%Y'))

        df = df.set_index('index')
        df = df.sort_index()
        self.series = df
        self._split_dataset(start_test_period)

    def _split_dataset(self, start_test_period):
        #train-test split dataset
        self.dataset_train = self.series[self.series['date'] < start_test_period]
        self.dataset_val = self.series[(self.series['date'] >= start_test_period)]
        