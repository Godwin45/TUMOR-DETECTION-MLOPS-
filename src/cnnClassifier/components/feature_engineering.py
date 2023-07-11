import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import FeatureConfig
import pandas as pd
from sklearn.model_selection import train_test_split

class FeatureEngineering:
    def __init__(self, config:FeatureConfig):
        self.config = config
        self.brain_df = None
        self.brain_df_train = None
        self.train = None
        self.test = None


    def preprocess_data(self):
        brain_df = pd.read_csv(self.config.brain_df)
        brain_df_train = brain_df.drop(columns=['patient_id'])
        brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))
        train,test = train_test_split(brain_df_train, test_size=0.15)

        train_file_path = os.path.join("artifacts/data_ingestion/Brain_MRI", "train.csv")
        test_file_path = os.path.join("artifacts/data_ingestion/Brain_MRI", "test.csv")

        train.to_csv(train_file_path, index=False)
        test.to_csv(test_file_path, index=False)

        print("The training and testing CSV files have been saved.")