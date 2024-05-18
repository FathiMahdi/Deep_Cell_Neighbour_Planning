from sklearn.model_selection import train_test_split
from csv_edit import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# delete unwanted columns
def DatasetPreProcessing(df,columns_head_list):

    try:

        df = df.drop(columns_head_list, axis='columns')
        

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
        train_dataset = train_dataset.batch(128*2)
        test_dataset = test_dataset.batch(128*2)


        # Return datasets with input shape and batch dimension
        return (train_dataset, train_target), (test_dataset, test_target), train_features.shape[1:]  
    
    except Exception as e:
        print("Cannot convert pandas dataframe to TensorFlow:", e)
        return None




def DatasetEncoding(pandas_data_frame, columns_head_list):

    try:

        encoded_df = pd.get_dummies(pandas_data_frame, columns=columns_head_list,dtype=int)
        
        print(encoded_df)

        return encoded_df
    
    except Exception as error:

        print("- Error in encoding dataset:", error)

        return None



# convert .csv dataset to pandas dataframe
def LoadDataSet(dataset_path):

    try:

        df = pd.read_csv(dataset_path)

        return df

    except  Exception as error:

        print("- Cannot load dataset!! ",error)

        return None

    


# Function to scale angles from 0-360 to -180 to 180
def scale_angle(angle):
    if angle > 180:
        return angle - 360
    else:
        return angle
    

# build tensorflow model
def BuildModle(input_shape):
    model = keras.Sequential() # create a new model
    model.add(keras.layers.Dense(64,activation="relu")) 
    model.add(keras.layers.Dense(64,activation="relu")) 
    model.add(keras.layers.Dense(32,activation="relu")) 
    model.add(keras.layers.Dense(16,activation="relu")) 
    model.add(keras.layers.Dense(8,activation="relu")) 
    model.add(keras.layers.Dense(4,activation="relu")) 
    model.add(keras.layers.Dense(2,activation="relu")) 
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=["accuracy"]) # model compilation
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


# value normalization
def dfDataNormalization(train_data, test_data):

    # Compute mean and standard deviation using the training dataset
    mean = tf.reduce_mean(train_data, axis=0)
    std = tf.math.reduce_std(train_data, axis=0)

    # Normalize the training dataset
    train_dataset_normalized = (train_data - mean) / std

    # Normalize the test dataset using the same mean and std
    test_dataset_normalized = (test_data - mean) / std

    return train_dataset_normalized, test_dataset_normalized



# Modify the main cell ID name and cell ID
find_replace_file('GUL_DS.csv', 'dataset.csv')


# for extracting 2G data only
# generate_2g_csv('dataset.csv', 'dataset.csv')

# load dataset
df = LoadDataSet('dataset.csv')


# remove unwanted columns
df = DatasetPreProcessing(df,['Cell ID', 'Main_Cell ID'] )


# angle scalling
df['Azimuth'] = df['Azimuth'].apply(scale_angle)
df['Main_Azimuth'] = df['Main_Azimuth'].apply(scale_angle)


# show dataframe
print(df)


# aply hot encoding
# encoded_pd = DatasetEncoding(df,['Main_Cell ID', 'Cell ID'])


# split data set
(train_data, train_target), (test_data, test_target), input_shape = ConverDFToTensorflowDSt('is_NBR', df)


# # implement data normalization
# train_data, test_data = dfDataNormalization(train_data,test_data)

print(train_data)
# # make Machine Learing model
# loaded_model = BuildModle(input_shape) // to rebuild model 

# Load the saved model
loaded_model = tf.keras.models.load_model("models/DNP.h5") # load saved model

loaded_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=["accuracy"])

# trainthe model
loaded_model.fit(train_data, epochs=1400, validation_data=test_data)

#save the model
loaded_model.save("models/DNP.h5")


# manual testing
predictions = loaded_model.predict(test_data)
print(predictions[0])
