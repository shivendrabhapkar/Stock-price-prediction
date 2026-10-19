# -*- coding: utf-8 -*-
"""
    Miscellaneous Functions for Regression File.
"""

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, chi2

try:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
except ImportError:
    QDA = None

try:
    from sklearn.model_selection import GridSearchCV, train_test_split
except ImportError:
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split

try:
    from Neural_Network import NeuralNet
except ImportError:
    NeuralNet = None

def load_dataset(path_directory, symbol): 
    """
        Import DataFrame from Dataset.
    """
    path = os.path.join(path_directory, symbol)
    out = pd.read_csv(path)
    if 'Unnamed: 0' in out.columns:
        out.drop('Unnamed: 0', axis=1, inplace=True)
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'])
        out.set_index('date', inplace=True)
    return [out]    

def count_missing(dataframe):
    """
    count number of NaN in dataframe
    """
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()
    
def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days
    """
    return_n = str(adjclose)[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = str(returns)[7:] + "RolMean" + str(n)
    dataframe[roll_n] = dataframe[returns].rolling(n).mean()

    exp_ma = str(returns)[7:] + "ExponentMovingAvg" + str(n)
    dataframe[exp_ma] = dataframe[returns].ewm(halflife=n).mean()
    
def mergeDataframes(datasets):
    """
        Merge Datasets into Dataframe.
    """
    return pd.concat(datasets)
    
def applyTimeLag(dataset, lags, delta):
    """
        apply time lag to return columns selected according to delta.
        Days to lag are contained in the lags list passed as argument.
        Returns a NaN free dataset obtained cutting the lagged dataset
        at head and tail
    """
    maxLag = max(lags)
    step = max(1, 2 * max(delta) - 1)
    columns = dataset.columns[::step]
    for column in columns:
        newcolumn = str(column) + str(maxLag)
        dataset[newcolumn] = dataset[column].shift(maxLag)

    return dataset.iloc[maxLag:-1, :].copy()

# CLASSIFICATION    
def prepareDataForClassification(dataset, start_test):
    """
    generates categorical to be predicted column, attach to dataframe 
    and label the categories
    """
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.loc[dataset.UpDown >= 0, 'UpDown'] = 'Up'
    dataset.loc[dataset.UpDown < 0, 'UpDown'] = 'Down'
    dataset.UpDown = le.fit_transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]    
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test    

def prepareDataForModelSelection(X_train, y_train, start_validation):
    """
    gets train set and generates a validation set splitting the train.
    The validation set is mandatory for feature and model selection.
    """
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]    
    
    X_val = X_train[X_train.index >= start_validation]    
    y_val = y_train[y_train.index >= start_validation]   
    
    return X, y, X_val, y_val
  
def performClassification(X_train, y_train, X_test, y_test, method, parameters={}):
    """
        Perform Classification with the help of several Algorithms.
    """
    print('Performing ' + method + ' Classification...')
    print('Size of train set: ', X_train.shape)
    print('Size of test set: ', X_test.shape)

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        KNeighborsClassifier(),
        SVC(C=10000),
        AdaBoostClassifier(**parameters),
        GradientBoostingClassifier(n_estimators=100),
    ]
    if QDA is not None:
        classifiers.append(QDA())

    scores = []
    for classifier in classifiers:
        scores.append(benchmark_classifier(classifier, X_train, y_train, X_test, y_test))

    print(scores)

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

# REGRESSION
def getFeatures(X_train, y_train, X_test, num_features):
    ch2 = SelectKBest(chi2, k=min(num_features, X_train.shape[1]))
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test

def performRegression(dataset, split, symbol, output_dir):
    """
        Performing Regression on Various algorithms
    """
    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0] * split))
    train, test = dataset.iloc[:index], dataset.iloc[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)

    out_params = (symbol, output_dir)
    output = dataset.columns[0]

    predicted_values = []

    classifiers = [
        RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        SVR(C=1000, kernel='rbf', epsilon=0.1, gamma='scale'),
        BaggingRegressor(random_state=42),
        AdaBoostRegressor(random_state=42),
        KNeighborsRegressor(),
        GradientBoostingRegressor(random_state=42),
    ]

    for classifier in classifiers:
        predicted_values.append(benchmark_model(classifier,
            train, test, features, output, out_params))

    print('-' * 80)
    mean_squared_errors = []
    r2_scores = []

    y_test_arr = test[output].values
    for pred in predicted_values:
        mse = mean_squared_error(y_test_arr, pred)
        r2 = r2_score(y_test_arr, pred)
        mean_squared_errors.append(mse)
        r2_scores.append(r2)

    return mean_squared_errors, r2_scores

def benchmark_model(model, train, test, features, output,
    output_params, *args, **kwargs):
    '''
        Performs Training and Testing of the Data on the Model.
    '''
    print('-' * 80)
    model_name = model.__class__.__name__.replace('Regressor', ' Regressor')
    print(model_name)

    symbol, output_dir = output_params

    X_train = train[features].values
    y_train = train[output].values
    X_test = test[features].values
    y_test = test[output].values

    model.fit(X_train, y_train, *args, **kwargs)
    predicted_value = model.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, color='g', ls='-', label='Actual Value')
    plt.plot(predicted_value, color='b', ls='--', label='Predicted Value')

    plt.xlabel('Test Instance Index')
    plt.ylabel('Price Value')
    plt.title(f"{symbol} - {model_name}")
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, str(symbol) + '_' + model_name + '.png'), dpi=100)
    plt.close()

    return predicted_value
