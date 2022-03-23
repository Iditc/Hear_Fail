# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
import shap
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


def train_randomforest_classifier(X_train, y_train):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf


def train_gradientboosting_classifier(X_train, y_train):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    return clf

def metrics_scoring(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    return precision, accuracy


def cm_display(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()


def roc_auc(X_test, y_test, clf, threshold):
    # define metrics
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    y_pred = [1 if x >= threshold else 0 for x in y_pred_proba]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def best_threshold(X_test, y_test):
    yhat = clf.predict_proba(X_test)  # predict probabilities
    yhat = yhat[:, 1]  # keep probabilities for the positive outcome only
    fpr, tpr, thresholds = roc_curve(y_test, yhat)  # calculate roc curves
    auc = roc_auc_score(y_test, yhat)
    print('Aus=' + str(auc))
    gmeans = np.sqrt(tpr * (1 - fpr))  # calculate the g-mean for each threshold
    ix = np.argmax(gmeans)  # locate the index of the largest g-mean
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')  # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.', label='Gradient Boosting Aus=' + str(auc))
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    pyplot.show()
    return thresholds[ix]


def change_threshold(X_test, y_test, clf):
    threshold = best_threshold(X_test, y_test)
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    y_pred = [1 if x >= threshold else 0 for x in y_pred_proba]
    return y_pred


def creatinine_th(df, threshold_h, threshold_l):
    df['creatinine_th_h'] = df['creatinine_phosphokinase'].apply(lambda x: 1 if x >= threshold_h else 0)
    df['creatinine_th_l'] = df['creatinine_phosphokinase'].apply(lambda x: 1 if threshold_l >= x else 0)
    return df


def ejection_fraction_th(df, threshold):
    df['ejection_fraction_th'] = df['ejection_fraction'].apply(lambda x: 1 if x >= threshold else 0)
    return df


def serum_sodium_th(df, threshold_h, threshold_l):
    df['serum_sodium_th_h'] = df['serum_sodium'].apply(lambda x: 1 if x >= threshold_h else 0)
    df['serum_sodium_th_l'] = df['serum_sodium'].apply(lambda x: 1 if threshold_l >= x else 0)
    return df


def platelets_th(df, threshold_h, threshold_l):
    df['thrombocytosis'] = df['platelets'].apply(lambda x: 1 if x >= threshold_h else 0)
    df['thrombocytopenia'] = df['platelets'].apply(lambda x: 1 if threshold_l >= x else 0)
    return df


def serum_creatinine_th(df, threshold_h, threshold_l):
    df['serum_creatinine_th_h'] = df['serum_creatinine'].apply(lambda x: 1 if x >= threshold_h else 0)
    df['serum_creatinine_th_l'] = df['serum_creatinine'].apply(lambda x: 1 if threshold_l >= x else 0)
    return df


def feature_generation(df):
    women_df = df[df['sex'] == 0]
    men_df = df[df['sex'] == 1]
    men_df = creatinine_th(men_df, threshold_h=308, threshold_l=39)
    women_df = creatinine_th(women_df, threshold_h=192, threshold_l=26)
    men_df = serum_creatinine_th(men_df, threshold_h=1.35, threshold_l=0.74)
    women_df = serum_creatinine_th(women_df, threshold_h=1.04, threshold_l=0.59)
    all = men_df.append(women_df)
    all = ejection_fraction_th(all, threshold=40)
    all = platelets_th(all, threshold_h=450000, threshold_l=150000)
    all = serum_sodium_th(all, threshold_h=135, threshold_l=145)
    return all


def predict(X_test, y_test, clf):
    y_pred = change_threshold(X_test, y_test, clf)
    precision = metrics_scoring(y_test, y_pred)
    return precision


# def feature_selection(df_all):
#     df_all =

if __name__ == '__main__':
    df_read = read_file('data/heart_failure_clinical_records_dataset.csv')
    print(df_read.columns)
    df_all = feature_generation(df_read)
    y = df_all[['DEATH_EVENT']].copy()
    X = df_all.drop(['DEATH_EVENT'], axis=1)
    class_names = ['DEATH_EVENT', 'No_DEATH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # clf = train_randomforest_classifier(X_train, y_train)
    clf = train_gradientboosting_classifier(X_train, y_train)
    precision_test, accuracy_test = predict(X_test, y_test, clf)
    precision_train, accuracy_train = predict(X_train, y_train, clf)
    print('precision:', precision_test)
    print('precision_t:', precision_train)
    print('accuracy:', accuracy_test)
    print('accuracy_t:', accuracy_train)
    # compute SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X.values, plot_type="bar", class_names=class_names, feature_names=X.columns, max_display=15)
    #shap.summary_plot(shap_values, X.values, feature_names=X.columns, max_display=11)



    y_pred = clf.predict(X_test)
    y_pred_t = clf.predict(X_train)
    precision = metrics_scoring(y_test, y_pred)
    precision_t = metrics_scoring(y_train, y_pred_t)
    print('precision:', precision)
    print('precision_t:', precision_t)
    y_pred = change_threshold(X_test, y_test, clf)
    precision = metrics_scoring(y_test, y_pred)
    print('precision:', precision)
    cm_display(y_test, y_pred)
    df_all.columns

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
