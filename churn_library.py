# library doc string


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from constant import rfc_model_path, lr_model_path, data_path, keep_cols, cat_columns, quant_columns, feature_imp_path
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def load_model_preds(pth_rfc, pth_lr) -> pd.Series:
    '''
    returns y_pred_train, y_pred_test for both the models RFC and LR
    input:
            pth: a path to both the models
    output:
            series: y_pred_train, y_pred_test(Predictions for both train and test datacoind )
    '''
    # Loading the model
    rfc_model = joblib.load(pth_rfc)
    lr_model = joblib.load(pth_lr)

    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    return rfc_model, lr_model, y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr


def import_data(pth) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    bank_data_df = pd.read_csv(pth, index_col=0)
    bank_data_df['Churn'] = bank_data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return bank_data_df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    bank_data_df = df.copy()

    # EDA for the attributes Churn, Customer_Age
    for var in ['Churn', 'Customer_Age']:
        plt.figure(figsize=(20,10))
        plt.xlabel(xlabel= var)
        plt.ylabel(ylabel= "Frequency")
        plt.title(f'This is a plot showing Histogram for {var}')
        bank_data_df[var].hist()
        plt.savefig(f'./images/eda/{var}_EDA.jpg')

    # EDA for Marital Status
    plt.figure(figsize=(20,10)) 
    bank_data_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/Marital_Status_EDA.jpg')

    # EDA of 'Total_Trans_Ct'
    plt.figure(figsize=(20,10)) 
    sns.histplot(bank_data_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/Total_Trans_Ct_EDA.jpg')

    # EDA (Heat Map)
    plt.figure(figsize=(20,10)) 
    sns.heatmap(bank_data_df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/Heat_Map.jpg')


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list,
        response=None) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    bank_data_df = df.copy()

    for var in category_lst:
        cat_columns_list = []
        gender_groups = bank_data_df.groupby(var).mean()['Churn']
        for val in bank_data_df[var]:
            cat_columns_list.append(gender_groups.loc[val])
        bank_data_df[var + "_Churn"] = cat_columns_list

    return bank_data_df


def perform_feature_engineering(df: pd.DataFrame, response: None , new_cols: list =keep_cols):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    bank_data_df = df.copy()
    y = bank_data_df.pop('Churn')
    X = bank_data_df[new_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Plotting the classification report by RFC
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/classification_report_RFC.jpg")

    # Plotting the classification report by LRC
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/classification_report_LRC.jpg")

def roc_plot(model_lr, model_rfc, X_test_data: pd.DataFrame, y_test_data: pd.Series):
    # Model evaluation using the saved models.
    lrc_plot = plot_roc_curve(model_lr, X_test_data, y_test_data)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(model_rfc, X_test_data, y_test_data, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve.jpg')

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # saving the plot
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Initializing the Models
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }
    # Training the models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    #saving the models trianed
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Loading the Model
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Model evaluation using the saved models.
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve.jpg')


if __name__ == "__main__":
    df_ = import_data(pth="./data/bank_data.csv")
    print(df_.shape)
    print(df_.columns)
    df_1 = encoder_helper(df=df_, category_lst=cat_columns)
    perform_eda(df = df_1)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df_1, response=None)
    rfc_model, lr_model, y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr = load_model_preds(
        pth_rfc=rfc_model_path, pth_lr=lr_model_path)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(model= rfc_model, X_data= X_test, output_pth= feature_imp_path)
    train_models(X_train, X_test, y_train, y_test)
    # roc_plot(
    #          lr_model,
    #          rfc_model,
    #          X_test,
    #          y_test)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
