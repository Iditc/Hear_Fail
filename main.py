# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score

import numpy as np


def read_file(file_name):
    read_df = pd.read_csv(file_name)
    return read_df


def xy_plot(df):
    plt.hist(df['serum_sodium'], bins=100)
    plt.ylabel('Count')
    plt.xlabel('Level of serum sodium in the blood (mEq/L)')
    plt.title("Sodium Distribution")
    #show()


def train_randomforestclassifier(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf


def metrics_scoring(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return precision


def cm_display(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()


def roc(X_test, y_test, clf):
    # define metrics
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = read_file('data/heart_failure_clinical_records_dataset.csv')
    print(df.columns)
    women_df = df[df['sex'] == 0]
    men_df = df[df['sex'] == 0]
    y = df[['DEATH_EVENT']].copy()
    X = df.drop(['DEATH_EVENT'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    clf = train_randomforestclassifier(X_train, X_test, y_train, y_test)
    y_pred = clf.predict(X_test)
    roc(X_test, y_test, clf)
    metrics_scoring(y_test, y_pred)
    cm_display(y_test, y_pred)


    xy_plot(women_df)
    xy_plot(men_df)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
