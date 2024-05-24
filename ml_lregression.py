from sklearn import datasets
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os # import OS for input output streaming
import time
import joblib
import matplotlib.pyplot as plt
import math # for math operations
import statistics as st # for statistics opration
from itertools import chain # converting 2D matrix into 1D vector
from csv_edit import *


def Normalization(X,Y):

    sc = StandardScaler()
    Label_Encoder = LabelEncoder()
    Y = Label_Encoder.fit_transform(Y)
    X = sc.fit_transform(X)
    return (X,Y)

def DatasetPreProcessing(df,columns_head_list):


    try:

        df = df.drop(columns_head_list, axis='columns')
        
        print(df)

        return df

    except Exception as error:

        print("Error while data preprocessing ",error)

        return None
    

def LoadDataSet(dataset_path):

    try:

        df = pd.read_csv(dataset_path)
        df = DatasetPreProcessing(df,['Main_Longitude','Longitude','Main_Latitude','Latitude', 'Cell ID', 'Main_Cell ID'] )
        X = df.iloc[:,1:-1].values
        Y = df['is_NBR'].values
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)


    except  Exception as error:

        print("- Cannot load dataset!! ",error)

    return (X, Y, df)



# def train_models(df):





# Modify the main cell ID name and cell ID
find_replace_file('GUL_DS.csv', 'dataset.csv')

# extract the 2G part
generate_2g_csv('dataset.csv', 'dataset.csv')


# load dataset
(X, Y, df) = LoadDataSet('dataset.csv')


# Normalization
(X, Y) = Normalization(X,Y)


# split the dataset
trn_fet, tst_fet, trn_leb, tst_leb = tts(X, Y, test_size=0.2)


svm_clf = svm.SVC(kernel='rbf')
svm_clf.fit(trn_fet, trn_leb)
joblib_file = "svm_model_test.pkl"
joblib.dump(svm_clf, joblib_file)
joblib_model = joblib.load(joblib_file) # load the model
start1 = time.time()
svm_prediction = joblib_model.predict(tst_fet)
print('The prediction value is: ',svm_prediction)
end1 = time.time()
print('\n SVM prediction response time = ', float(end1-start1))
print('\n accuracy of SVM is: ',accuracy_score(tst_leb,svm_prediction))
print(classification_report(tst_leb,svm_prediction))
print(confusion_matrix(tst_leb,svm_prediction))
plt.matshow(confusion_matrix(tst_leb,svm_prediction),fignum='SVM confusion matrix' ,cmap=plt.cm.gray)
score = joblib_model.score(tst_fet, tst_leb)
print("Test score: {0:.2f} %".format(100 * score))
# train_models(df)