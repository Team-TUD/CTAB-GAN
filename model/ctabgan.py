import pandas as pd
import time
# Used for pre/post-processing of the input/generated data
from model.pipeline.data_preparation import DataPrep 
# Model class for the CTABGANSynthesizer
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution 
    6) integer_columns -> list of numeric column names without floating numbers  
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data 
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data 

    """

    def __init__(self,
                 raw_csv_path = "Real_Datasets/Adult.csv",
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs = 1):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(epochs = epochs)
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


    def generate_samples(self):
        
        sample = self.synthesizer.sample(len(self.raw_df)) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
