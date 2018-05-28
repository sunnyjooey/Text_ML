# Reference: github.com/Computational-Content-Analysis-2018/lucem_illud.git
# Reference: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn import metrics


def visualize_features(cols, forest, n):
    """
    Visualize the most important features
    Inputs: already fit classifier `forest`, top `n` features
    """
    # Get important features
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    ind = np.argsort(importances)[::-1]
    indices = ind[:n]
    
    xlab = []
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(n):
        print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))
        xlab.append(cols[indices[f]])
        
    c = ['orangered','darkorange']
    if n % 2 == 0:
        colors = c * int(n/2)
    else:
        colors = (c * int((n+1)/2))[:n]
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(n), importances[indices],
           color=colors, yerr=std[indices], align="center")
    plt.xticks(range(n), xlab, rotation='vertical')
    plt.xlim([-1, n])
    plt.show()


def plotConfusionMatrix(clf, predictions, y_test):
    """
    Plot the confustion matrix of predicted val v. actual
    Prints a graph (matrix)
    """
    mat = metrics.confusion_matrix(predictions, y_test)
    seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=y_test.unique(), yticklabels=y_test.unique())
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title("Confusion Matrix")
    plt.show()
    plt.close()

    
def plotMultiROC(clf, predictions, X_test, y_test):
    """
    Plot the ROC curve per label.
    Prints a graph
    """
    # get probabiltiess
    classes = clf.classes_
    try:
        probs = clf.predict_proba(X_test)
    except AttributeError:
        print("The {} classifier does not apear to support prediction probabilties.".format(type(clf)))
        return
    
    # setup axis for plotting
    fig, ax = plt.subplots(figsize = (10,10))

    # get AUC values
    aucVals = []
    for classIndex, className in enumerate(classes):        
        truths = [1 if c == className else 0 for c in y_test]
        predict = [1 if c == className else 0 for c in predictions]
        scores = probs[:, classIndex]

        # get the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(truths, scores)
        auc = metrics.auc(fpr, tpr)
        aucVals.append(auc)

        # plot the label 
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(str(className).split(':')[0], auc))

    # formatting
    ax.set_title('Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.show()
    plt.close()

    
def evaluateClassifier(clf, predictions, y_test):
    """
    Get basic stats of the classifier
    Returns a table, one row for each label
    """
    classes = []
    results = {
        'Error_Rate' : [],
        'AUC' : [],
        'Precision' : [],
        'Average_Precision' : [],
        'Recall' : [],
        }

    for cat in set(y_test):
        preds = [True if (c == cat) else False for c in predictions]
        acts = [True if (c == cat) else False for c in y_test]
        classes.append(cat)
        results['AUC'].append(metrics.roc_auc_score(acts, preds))
        results['Average_Precision'].append(metrics.average_precision_score(acts, preds))
        results['Precision'].append(metrics.precision_score(acts, preds))
        results['Recall'].append(metrics.recall_score(acts, preds))
        results['Error_Rate'].append(1 -  metrics.accuracy_score(acts, preds))
    df = pd.DataFrame(results, index=classes)
    df.index.rename('Label', inplace=True)
    return df