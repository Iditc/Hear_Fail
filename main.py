# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = read_file('data/heart_failure_clinical_records_dataset.csv')
    women_df = df[df['sex'] == 0]
    men_df = df[df['sex'] == 0]
    xy_plot(women_df)
    xy_plot(men_df)
    print(df.columns)
    print ('hi')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
