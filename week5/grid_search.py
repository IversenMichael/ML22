from sklearn.datasets import load_wine, load_breast_cancer
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_covtype


def show_result(clf):
    df = pd.DataFrame(clf.cv_results_)
    df = df.sort_values('mean_test_score', ascending=False)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print('best parameter found', clf.best_params_)


w_data = load_wine()
wine_data = w_data.data
wine_labels = w_data.target

# grid search validation
reg_parameters = {'max_depth': list(range(1, 10)), 'min_samples_split': list(range(1, 10))}  # dict with all parameters we need to test
clf = GridSearchCV(DecisionTreeClassifier(), reg_parameters, cv=3, return_train_score=True)
clf.fit(wine_data, wine_labels)
# code for showing the result
show_result(clf)

cancer_data = load_breast_cancer()
c_data = cancer_data.data
c_labels = cancer_data.target


def decisiontree_model_selection(train_data, labels):
    clf = None
    ### YOUR CODE HERE
    ### END CODE
    return clf
###
clf = decisiontree_model_selection(c_data, c_labels)
bt = show_result(clf)