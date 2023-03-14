import os
import logging
from constant import cat_columns
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, import_data):
	'''
	test perform eda function
	'''
	try:
		dataframe = import_data("./data/bank_data.csv")
		perform_eda(dataframe)
		assert os.listdir('./images/eda') != []
		logging.info("Testing perform_eda: The folder is not empty(perform_eda ran successfully)")
	except AssertionError as err:
		logging.error("Testing perform_eda: The folder is empty(perform_eda didnt ran successfully)")
		raise err


def test_encoder_helper(encoder_helper, import_data):
	'''
	test encoder helper
	'''
	dataframe = import_data('./data/bank_data.csv')
	encoder_helper(dataframe, cat_columns)



def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import(import_data)








