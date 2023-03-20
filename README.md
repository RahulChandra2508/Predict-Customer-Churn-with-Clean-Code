# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project **Predict Customer Churn** is used to predict the churn probablity i.e.,
credit card customers that are most likely to churn. The completed project will include a 
Python package for a machine learning project that follows coding (PEP8) and engineering 
best practices for implementing software (modular, documented, and tested). 
The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

### setting up the project
    Before we execute the projecv we need to set the environment.
    We can do this by installing all the required libraries using requirements.txt.
    "pip install -r requirements_py36.txt" for python version 3.6
    "pip install -r requirements_py36.txt" for python version 3.8

## Files and data description
# churn_library:
churn_library files contains the functions that are used in conjunction
to have al the preprocessing and training the model which is used
to predict the churn.

# churn_script_logging_and_tests:

churn_script_logging_and _tests files Contain unit tests for the churn_library.py that are used to
test the functions within the churn_library which is used to predict the churn of
credit card customers.

The goal of test functions is to checking the returned items aren't empty or folders where results 
should land have results after the function has been run.

# Data floder:
Data folder contains the required data in .csv format that is used to train the ML model

# Images:
Images folder contains the eda and result images saved when churn_library is ran.

# logs:
logs the information or errors while testing the ML functions when churn_script_logging_and_tests is ran.

## Running Files
# churn_library:
ipython churn_library.py
# churn_script_logging_and_tests:
ipython churn_script_logging_and_tests.py

