import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection


class DataPrep(object):
    
    """
    Data preparation class for pre-processing input data and post-processing generated data

    Variables:
    1) raw_df -> dataframe containing input data
    2) categorical -> list of categorical columns
    3) log -> list of skewed exponential numerical columns
    4) mixed -> dictionary of "mixed" column names with corresponding categorical modes 
    5) integer -> list of numeric columns without floating numbers
    6) type -> dictionary of problem type (i.e classification/regression) and target column
    7) test_ratio -> ratio of size of test to train dataset

    Methods:
    1) __init__() -> instantiates DataPrep object and handles the pre-processing steps for feeding it to the training algorithm
    2) inverse_prep() -> deals with post-processing of the generated data to have the same format as the original dataset


    """


    def __init__(self, raw_df: pd.DataFrame, categorical: list, log:list, mixed:dict, integer:list, type:dict, test_ratio:float):
        
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []

        # Spliting the input data to obtain training dataset
        target_col = list(type.values())[0]
        y_real = raw_df[target_col]
        X_real = raw_df.drop(columns=[target_col])
        X_train_real, _, y_train_real, _ = model_selection.train_test_split(X_real ,y_real, test_size=test_ratio, stratify=y_real,random_state=42)        
        X_train_real[target_col]= y_train_real

        # Replacing empty strings with na if any and replace na with empty
        self.df = X_train_real
        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')
        
        # Dealing with empty values in numeric columns by replacing it with -9999999 and treating it as categorical mode 
        all_columns= set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)
        
        for i in relevant_missing_columns:
            if i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x )
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):   
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_columns[i] = [-9999999]
        
        # Dealing with skewed exponential numeric distributions by applying log transformation
        if self.log_columns:
            for log_column in self.log_columns:
                # Value added to apply log to non-positive numeric values
                eps = 1 
                # Missing values indicated with -9999999 are skipped
                lower = np.min(self.df.loc[self.df[log_column]!=-9999999][log_column].values) 
                self.lower_bounds[log_column] = lower
                if lower>0: 
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x) if x!=-9999999 else -9999999)
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x+eps) if x!=-9999999 else -9999999) 
                else:
                    # Negative values are scaled to become positive to apply log
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x-lower+eps) if x!=-9999999 else -9999999)
        
        # Encoding categorical column using label encoding to assign each category within a column with an integer value
        for column_index, column in enumerate(self.df.columns):
            
            if column in self.categorical_columns:        
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)
            
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]
        
        super().__init__()
        
    def inverse_prep(self, data, eps=1):
        
        # Converting generated data into a dataframe and assign column names as per original dataset
        df_sample = pd.DataFrame(data,columns=self.df.columns)
        
        # Reversing the label encoding assigned to categorical columns according to the original dataset 
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = df_sample[self.label_encoder_list[i]["column"]].astype(int)
            df_sample[self.label_encoder_list[i]["column"]] = le.inverse_transform(df_sample[self.label_encoder_list[i]["column"]])

        # Reversing log by applying exponential transformation with appropriate scaling for non-positive numeric columns 
        # -9999999 used to denote missing values are similarly ignored
        if self.log_columns:
            for i in df_sample:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound>0:
                        df_sample[i].apply(lambda x: np.exp(x) if x!=-9999999 else -9999999) 
                    elif lower_bound==0:
                        df_sample[i] = df_sample[i].apply(lambda x: np.ceil(np.exp(x)-eps) if ((x!=-9999999) & ((np.exp(x)-eps) < 0)) else (np.exp(x)-eps if x!=-9999999 else -9999999))
                    else: 
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x)-eps+lower_bound if x!=-9999999 else -9999999)
        
        # Rounding numeric columns without floating numbers in the original dataset
        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column]= (np.round(df_sample[column].values))
                df_sample[column] = df_sample[column].astype(int)

        # Converting back -9999999 and "empty" to na
        df_sample.replace(-9999999, np.nan,inplace=True)
        df_sample.replace('empty', np.nan,inplace=True)

        return df_sample
