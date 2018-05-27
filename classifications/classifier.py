import pandas as pd
import pickle
import time
import plotter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def run_models(feat_types, clfs):
    """
    Given dictionary of `feat_types` (referring to dataframes with different 
    feature types) and dictionary of `clfs` (values are classifiers),
    print classifier statistics and graphs
    """
    for feat_key, feat_val in feat_types.items():
        X_train = pd.read_pickle('./data/{}/X_train_ft.pkl'.format(feat_val))
        X_test = pd.read_pickle('./data/{}/X_test_ft.pkl'.format(feat_val))
        y_train = pd.read_pickle('./data/{}/y_train.pkl'.format(feat_val))
        y_test = pd.read_pickle('./data/{}/y_test.pkl'.format(feat_val))
        print(feat_key)
    
        for clf_key, clf in clfs.items():
            print(clf_key)
            start_time = time.time()
            clf_fit = clf.fit(X_train, y_train)
            predictions = clf.predict(X_test) 
            display(plotter.evaluateClassifier(clf, predictions, y_test))
            plotter.plotConfusionMatrix(clf, predictions, y_test)
            plotter.plotMultiROC(clf, predictions, X_test, y_test)
            print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
            print()
        print()


def run_ensemble(feat_types, clf, feat_viz=False, n=None):
    """
    Given a `clf` ensemble classifier, print classifier statistics and graphs
    If `feat_viz`, print most important `n` features
    Returns dictionary of objects related to features   
    """
    objs = {}
    for feat_key, feat_val in feat_types.items():
        X_train = pd.read_pickle('./data/{}/X_train_ft.pkl'.format(feat_val))
        X_test = pd.read_pickle('./data/{}/X_test_ft.pkl'.format(feat_val))
        y_train = pd.read_pickle('./data/{}/y_train.pkl'.format(feat_val))
        y_test = pd.read_pickle('./data/{}/y_test.pkl'.format(feat_val))
        obj = pickle.load(open('./data/{}/obj.pkl'.format(feat_val), 'rb'))
        objs[feat_val] = obj
        print(feat_key)
        
        # Fit
        clf_fit = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test) 
        display(plotter.evaluateClassifier(clf, predictions, y_test))
        plotter.plotConfusionMatrix(clf, predictions, y_test)
        plotter.plotMultiROC(clf, predictions, X_test, y_test)
                
        # Visualize
        if feat_viz:
            plotter.visualize_features(X_train.columns, clf_fit, n)
        print()
    return objs


