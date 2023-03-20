'''
Author: Rahul

Date of Modification: 15-03-2023

Description:
churn_script_logging_and _tests files contains the test functions that are used to
test the functions within the churn_library which is used to predict the churn of
credit card customers.
'''

import os
import pytest
import logging
import churn_library as cls
from constant import cat_columns, keep_cols

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w+',
    format='%(name)s - %(levelname)s - %(message)s')

## unable to fixture so went into calling the import in every function.
# @pytest.fixture 
# def import_data_df():
#     dataframe =  cls.import_data("./data/bank_data.csv")
#     logging.info("import_data: SUCCESS")
#     return dataframe


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info("Testing import_data: The file has data within it.")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, import_data):
    '''
    test perform eda function
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        perform_eda(dataframe)
        assert os.listdir('./images/eda') != []
        logging.info(
            "Testing perform_eda: The folder is not empty(perform_eda ran successfully)")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The folder is empty(perform_eda didnt ran successfully)")
        raise err


def test_encoder_helper(encoder_helper, import_data, cat_vars):
    '''
    test encoder helper
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe_transformed = encoder_helper(dataframe, cat_vars)
        assert dataframe_transformed.shape[1] == 27
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: encoder_helper is not working properly")
        raise err

    try:
        assert dataframe_transformed.shape[0] > 0
        assert dataframe_transformed.shape[1] > 0
        logging.info("Testing encoder_helper: The transformed data looks good")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The transformed data doesn't appear to have rows and columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, import_data, encoder_helper, cat_vars, new_cols):
    '''
    test perform_feature_engineering
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe_transformed = encoder_helper(dataframe, cat_vars)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe_transformed, new_cols)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: perform_feature_engineering is not working properly")
        raise err

    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) != 0
        assert len(y_test) != 0
        logging.info(
            "Testing perform_feature_engineering: The data split looks good")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The data split doesn't appear to meet the criteria of train test split")
        raise err


def test_train_models(train_models, perform_feature_engineering, import_data, encoder_helper, cat_vars, new_cols):
    '''
    test train_models
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe_transformed = encoder_helper(dataframe, cat_vars)
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            dataframe_transformed, new_cols)
        train_models(X_train, X_test, y_train, y_test)
        assert os.listdir('./models') != []
        assert os.listdir('./images/results') != []
        logging.info(
            "Testing test_train_models: The folder is not empty(perform_eda ran successfully)")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: The folder is empty(perform_eda didnt ran successfully)")
        raise err
    

if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda, cls.import_data)
    test_encoder_helper(cls.encoder_helper, cls.import_data, cat_columns)
    test_perform_feature_engineering(cls.perform_feature_engineering, cls.import_data, cls.encoder_helper, cat_columns, keep_cols)
    test_train_models(cls.train_models, cls.perform_feature_engineering, cls.import_data, cls.encoder_helper, cat_columns, keep_cols)
