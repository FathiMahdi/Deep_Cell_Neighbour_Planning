from tensorflow.keras.datasets import boston_housing
import pandas as pd
import numpy as np


def DatasetPreProcessing(df,columns_head_list):



    try:

        df = df.drop(columns_head_list, axis='columns')
        
        print(df)

        return df

    except Exception as error:

        print("Error while data preprocessing ",error)

        return None

    


def DatasetEncoding(pandas_data_frame, columns_head_list):

    try:

        encoded_df = pd.get_dummies(pandas_data_frame, columns=['Main_Cell ID'])
        
        return encoded_df
    
    except Exception as error:

        print("- Error in encoding dataset:", error)

        return None



def LoadDataSet(dataset_path):

    try:

        df = pd.read_csv(dataset_path)

        print(df)

    except  Exception as error:

        print("- Cannot load dataset!! ",error)

    return df
    
# 
def BuildModle():

    model = keras.Sequential() # create a new model
    model.add(keras.layers.Dense(64,activation="relu")) # add one layer whit 64 neuron 
    model.add(keras.layers.Dense(64,activation="relu"))# add one layer whit 64 neuron 
    model.add(keras.layers.Dense(1)) # add output layer
    model.compile(optimizer='rmsprop',loss='mse',metrics=["mae"]) # model compilation
    return model


# Data normaliztion function
def DataNormalization(train_data, test_data):

    # train_data normalization
    mean  = train_data.mean(axis=0)
    train_data -= mean

    std  = train_data.std(axis=0)
    train_data /= std

    # test data normalization
    mean  = test_data.mean(axis=0)
    test_data -= mean

    std  = test_data.std(axis=0)
    test_data /= std

    return train_data, test_data



# load dataset
df = LoadDataSet('dataset.csv')

df = DatasetPreProcessing(df,['Main_Longitude','Longitude','Main_Latitude','Latitude'])

# aply hot encoding
#encoded_pd = DatasetEncoding(df,['Main_Cell ID', 'Cell ID'])


# # get the house pricing data set
# (train_data, train_target), (test_data, test_target) = boston_housing.load_data()

# # implement data normalization
# train_data, test_data = DataNormalization(train_data,test_data)

# # make Machine Learing model
# model = BuildModle()

# # trainthe model
# model.fit(train_data, train_target)
