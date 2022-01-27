"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 raw_csv_path = "Real_Datasets/Adult.csv",
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs = 1,
                 batch_size=500,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(epochs = epochs, batch_size = batch_size, class_dim = class_dim, random_dim = random_dim,
                                              num_channels = num_channels, l2scale = l2scale)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):

        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"],
        mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, num_samples):

        sample = self.synthesizer.sample(num_samples)
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
