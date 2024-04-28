from sklearn.model_selection import train_test_split
from csv_edit import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
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



def ConverDFToTensorflowDSt(output, df):
    
    try:
        target = df.pop(output)
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df)
        # Split the dataset into training and testing sets
        train_data, test_data, train_target, test_target = train_test_split(normalized_features, target, test_size=0.2, random_state=42)
        # Convert features and targets to float32
        train_features = tf.constant(train_data, dtype=tf.float32)
        test_features = tf.constant(test_data, dtype=tf.float32)
        train_target = tf.constant(train_target.values, dtype=tf.float32)
        test_target = tf.constant(test_target.values, dtype=tf.float32)
        # Combine features and targets into tuples
        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_target))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_target))
        # Add batch dimension to datasets
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)
        # Return datasets with input shape and batch dimension
        return (train_dataset, train_target), (test_dataset, test_target), train_features.shape[1:]  
    
    except Exception as e:
        print("Cannot convert pandas dataframe to TensorFlow:", e)
        return None


      

def ConverDFToTensorflowDS(output,df):

    target = df.pop(output)

    try:

        # Split the dataset into training and testing sets
        train_data, test_data, train_target, test_target = train_test_split(df, target, test_size=0.2, random_state=42)


        # Create TensorFlow datasets for training and testing
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_target.values))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data.values, test_target.values))

        # Print some samples from the datasets
        print("Training dataset samples:")
        for features, targets in train_dataset.take(5):
            print('Features: {}, Target: {}'.format(features, targets))

        print("Testing dataset samples:")
        for features, targets in test_dataset.take(5):
            print('Features: {}, Target: {}'.format(features, targets))
        

        return train_dataset, test_dataset
    
    except Exception as e:

        print("Cannot conver pandas dataframe to tensorflow: ",e)






def DatasetEncoding(pandas_data_frame, columns_head_list):

    try:

        encoded_df = pd.get_dummies(pandas_data_frame, columns=columns_head_list,dtype=int)
        
        print(encoded_df)

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
def BuildModle(input_shape):

    model = keras.Sequential() # create a new model
    model.add(keras.layers.Dense(64,activation="relu",input_shape=input_shape)) # add one layer whit 64 neuron 
    model.add(keras.layers.Dense(64,activation="relu"))# add one layer whit 64 neuron 
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


def dfDataNormalization(train_data, test_data):

    # Compute mean and standard deviation using the training dataset
    mean = tf.reduce_mean(train_data, axis=0)
    std = tf.math.reduce_std(train_data, axis=0)

    # Normalize the training dataset
    train_dataset_normalized = (train_data - mean) / std

    # Normalize the test dataset using the same mean and std
    test_dataset_normalized = (test_data - mean) / std

    return train_dataset_normalized, test_dataset_normalized














# Modify the mane cell ID name and cell ID
find_replace_file('GUL_DS.csv', 'dataset.csv')

# load dataset
df = LoadDataSet('dataset.csv')


df = DatasetPreProcessing(df,['Main_Longitude','Longitude','Main_Latitude','Latitude'])

# aply hot encoding
encoded_pd = DatasetEncoding(df,['Main_Cell ID', 'Cell ID'])


# convert df to tensorflow dataset
#tf_dataset = ConverDFToTensorflowDS('is_NBR',encoded_pd)


# # get the house pricing data set
#(train_data, train_target), (test_data, test_target) = tf_dataset.load_data()

(train_data, train_target), (test_data, test_target), input_shape = ConverDFToTensorflowDSt('is_NBR', encoded_pd)


# # implement data normalization
#train_data, test_data = dfDataNormalization(train_data,test_data)


# # make Machine Learing model
model = BuildModle(input_shape)

# # trainthe model
#model.fit(train_data, train_target)
model.fit(train_data, epochs=100, validation_data=test_data)

#save the model
model.save("models/DNP.h5")

