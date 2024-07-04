import sys 
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender',
                                    'race/ethnicity',
                                    'parental level of education', 
                                    'lunch', 
                                    'test preparation course'
                                    ]
            num_pipeline=Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy="median")),
                   ("scaler",StandardScaler())
               ])
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
                    ("encoder",OneHotEncoder())
                ]
            )
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [("num_transformation",num_pipeline,numerical_features),
                 ("categorical_transformation",cat_pipeline,categorical_features)]
            )
            logging.info("Tranformation completed")

            return preprocessor

         
        except Exception as e:
            raise custom_exception(e,sys)
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Loading train and test data")

            preporocessor=self.get_data_transformer()

            target_column_name="math score"

            x_train=train_df.drop(target_column_name,axis=1)
            y_train=train_df[target_column_name]

            x_test=test_df.drop(target_column_name,axis=1)
            y_test=test_df[target_column_name]

            x_train_transformed=preporocessor.fit_transform(x_train)
            x_test_transformed=preporocessor.transform(x_test)

            train_arr=np.c_[x_train_transformed,np.array(y_train)]
            test_arr=np.c_[x_test_transformed,np.array(y_test)]

            logging.info("Saved preprocesing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preporocessor
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise custom_exception(e,sys)