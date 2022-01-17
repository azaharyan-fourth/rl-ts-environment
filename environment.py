import torch
import pandas as pd
import numpy as np
from forecasting_models.xgboost_model import XGBoostModel
from utils import parse_command_args, get_json_params
from dataset import Dataset
from gym import Env
from spaces.evenly_spaced import EvenlySpaced
from sklearn.metrics import mean_squared_error

class Environment(Env):

    def __init__(self, 
                dataset,
                target, 
                labor_feature,
                number_actions,
                start_action,
                stop_action,
                target_model_params,
                labor_model_params,
                cost_labor,
                window):

        self.dataset = dataset
        self.window = window
        self.action_space = EvenlySpaced(start_action, stop_action, number_actions)
        self.t = window #timestep of the series

        self.target = target
        self.labor_feature = labor_feature

        if target_model_params is not None:
            self.model_sales = XGBoostModel(**target_model_params)
        else:
            self.model_sales = XGBoostModel(n_estimators=100, learning_rate=0.1)

        if labor_model_params is not None:
            self.model_hours = XGBoostModel(**labor_model_params)
        else:
            self.model_hours = XGBoostModel(n_estimators=100, learning_rate=0.1)

        self.cost_hour = cost_labor

        self.device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

    def get_state(self, index=None, is_test=False):
        """ 
            Get the current state of the environment

            Args:
            index (int):
            is_test (bool): 
        """

        if index is None:
            index = self.t
        
        state = None

        if is_test:
            state = self.dataset.dataset_val.loc[index-self.window:index].copy()
        else:
            state = self.dataset.dataset_train.iloc[index-self.window:index+1].copy()

        return state

    def step(self, action, is_test=False):
        """ 
            Make step in the environment and return the next state and the reward
            return (nex_state, reward)

            Args:
            action_idx (int): index of the selected action
            is_test (bool): mode of the agent

            Returns:
            next_state
            reward (float): 
            done (bool): True when it is the last element of the dataset
                        (i.e. the series)
        """

        current_state = self.get_state(is_test=is_test)

        reward = self._apply_action_get_reward(action, current_state)
        self.t += 1

        done = self.t == len(self.dataset.dataset_train)-1
        next_state = self.get_state(is_test=is_test)

        if done:
            next_state = None

        return next_state, reward, done

    def reset(self, index=None):
        '''
        Resets the environment, i.e. moves the counter t to the 
        index with offset of size `window`

        Args:
        index (int): 
        '''
        if index != None:
            self.t = index + self.window
        else:   
            self.t = self.window

    def iter_dataset(self, train: bool = True):
        '''
        Wrapper for iteration of a dataset

            Args:
                train (bool): Shows if we should iterate the train dataset

        '''
        if train:
            dataset_to_iterate = self.dataset.dataset_train
        else:
            dataset_to_iterate = self.dataset.dataset_val

        for value in dataset_to_iterate[self.window+1:].iterrows():
            yield value
            
    def transform_data_for_nn(self, df) -> torch.Tensor:
        '''
        Drop unnecessary columns for the NN and transform the DataFrame to Tensor 

            Args:
                df (pd.DataFrame): DataFrame of the data

            Returns:
                data (torch.Tensor)
        '''

        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)
            
        data = torch.tensor(df.values, dtype=torch.float, device=self.device)

        return data

    def get_predicted_sales(self, current_state: pd.DataFrame) -> float:
        '''
            Get forecasted sales for the current state

            Args:
                current_state (DataFrame): the current state of the environment, which
                            consists of the current day + the context window

            Returns:
                prediction(float): 
        '''
        test = self.model_sales.create_features(current_state)
        test.drop(self.target, axis=1, inplace=True)
        prediction = self.model_sales.test(test.tail(1))[0]
        return prediction

    def get_predicted_labor(self, 
                            current_state: pd.DataFrame, 
                            forecast_sales=None) -> float:
        '''
            Get forecast of the feature for the current state

            Args:
                current_state(DataFrame):
                forecast_sales (float): if it is not None we should use this as the target
                                for the current day

            Returns:
                prediction(float): prediction of the labor rounded to 1 decimal place
        '''
        state = current_state.copy()
        if forecast_sales != None:
            state.loc[-1, self.target] = forecast_sales

        test = self.model_hours.create_features(state)
        test.drop(self.labor_feature, axis=1, inplace=True)
        prediction = self.model_hours.test(test.tail(1))[0]
        return np.float64(round(prediction,1))

    def train_environment_and_evaluate(self):
        """ 
            Train helper forecasting models of the environment
        """

        # Create features and train XGBoost models
        X_train, y_train = self.model_sales.create_features(self.dataset.dataset_train, label='sales')
        X_test, y_test = self.model_sales.create_features(self.dataset.dataset_val, label='sales')

        self.model_sales.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

        X_train_hours, y_train_hours = self.model_sales.create_features(self.dataset.dataset_train, label=self.labor_feature)
        X_test_hours, y_test_hours = self.model_sales.create_features(self.dataset.dataset_val, label=self.labor_feature)

        self.model_hours.train(X_train_hours, y_train_hours, 
                            eval_set=[(X_train_hours, y_train_hours), (X_test_hours, y_test_hours)])

        # remove 1 for the date column
        self.n_observation_space = len(X_train.columns)+1

        #Evaluate the XGBoost models and output the results
        wape_train = self.predict_and_evaluate(X_train, y_train)
        wape_test = self.predict_and_evaluate(X_test, y_test)

        rmse_labor_train = self.predict_and_evaluate(X_train_hours,
                                                    y_train_hours,
                                                    is_target=False,
                                                    metric='rmse')
        rmse_labor_test = self.predict_and_evaluate(X_test_hours,
                                                    y_test_hours,
                                                    is_target=False, 
                                                    metric='rmse')

        print(f"Target train WAPE: {wape_train}")                    
        print(f"Target test WAPE: {wape_test}")
        print(f"Labor train RMSE: {rmse_labor_train}")
        print(f"Labor test RMSE: {rmse_labor_test}")
        

    def predict_and_evaluate(self, 
                            X: pd.DataFrame,
                            y: pd.Series, 
                            is_target=True, metric='wape'):
        '''
        Evaluate the performance of the trained XGBoost models

        Args:
        X (DataFrame): data frame to get predictions for
        y (Series): true labels of the dataset
        is_target (bool): used to choose which XGBoost model to use
        metric (str): the evaluation metric to be used

        Returns:
        score (float): the score according to the specified metric 
        '''
        if is_target:
            pred = self.model_sales.test(X)
        else:
            pred = self.model_hours.test(X)

        if metric == 'wape':
            score = (abs(y - pred)).sum() / y.sum()
        elif metric == 'rmse':
            score = mean_squared_error(pred, y, squared=False)
        return score

    

    def _apply_action_get_reward(self, action, state):
        """ Apply action and pass the resulted reward

            Args:
            action_idx (int): index of the selected action in the action space
            state (pd.DataFrame): current state
        """

        action_value = torch.tensor(self.action_space[action])
        hours_actual = state.loc[self.t][self.labor_feature]
        sales_actual = state.loc[self.t][self.target]

        forecast_no_action = self.get_predicted_sales(state)
        forecast_labor_no_action = self.get_predicted_labor(state, forecast_sales=forecast_no_action)


        #apply action
        #state.loc[self.t, 'HoursWorked'] += action_value.numpy()
        forecast_labor = self.get_predicted_labor(state)
        state.loc[self.t, self.labor_feature] = forecast_labor+action_value.cpu().numpy()
        forecast_action = self.get_predicted_sales(state)

        forecast_action = self._fix_forecasts_minmax(forecast_action, forecast_no_action, action_value)
        hours = state.loc[self.t][self.labor_feature]

        forecast_profit = forecast_action - \
            state.loc[self.t][self.labor_feature]*self.cost_hour

        actual_profit = forecast_no_action - forecast_labor_no_action*self.cost_hour

        #(forecasts with action-labour with action) - (forecast with 0 - labour with 0)
        reward = forecast_profit - actual_profit

        return reward


    def _fix_forecasts_minmax(self, forecast_action, forecast_noaction, action):
        """ Fix forecasts, s.t. those for decreased labour do not exceed the one for
            no action and so on.

            Args:
            forecast_action (decimal): forecasted sales with applied action
            forecast_noaction (decimal): forecasted sales for no applied action (0)
            action (int): value of the selected action (not index)
        """

        if action > 0:
            forecast_action = max(forecast_action, forecast_noaction)
        elif action < 0:
            forecast_action = min(forecast_action, forecast_noaction)

        return forecast_action

if __name__ == '__main__':
    args = parse_command_args()
    
    target_params = get_json_params(args.target_params)
    labor_params = get_json_params(args.labor_params)

    dataset = Dataset(args.data_path, args.start_test_period)
    env = Environment(dataset,
                        args.target, 
                        args.labor_feature,
                        number_actions=int(args.number_actions),
                        start_action=float(args.start_action),
                        stop_action=float(args.stop_action),
                        target_model_params=target_params,
                        labor_model_params=labor_params,
                        cost_labor=args.cost_labor,
                        window=args.window_size)

    env.train_environment_and_evaluate()
