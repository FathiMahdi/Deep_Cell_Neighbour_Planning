from sklearn.model_selection import train_test_split
from csv_edit import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

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

        target = df[output]
        df.drop(columns=[output], inplace=True)

        # Normalize features
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse_output=False)
        normalized_features = scaler.fit_transform(df)
        encoded_target = encoder.fit_transform(target.values.reshape(-1, 1))  # Reshape target to 2D array

        # Split the dataset into training and testing sets
        train_data, test_data, train_target, test_target = train_test_split(normalized_features, encoded_target, test_size=0.2, random_state=42)
 

        # # Convert features and targets to float32
        # train_features = tf.constant(train_data, dtype=tf.float32)
        # test_features = tf.constant(test_data, dtype=tf.float32)
        # train_target = tf.constant(train_target.values, dtype=tf.float32)
        # test_target = tf.constant(test_target.values, dtype=tf.float32)


        train_features = tf.convert_to_tensor(train_data, dtype=tf.float32)
        test_features = tf.convert_to_tensor(test_data, dtype=tf.float32)
        train_target = tf.convert_to_tensor(train_target, dtype=tf.float32)
        test_target = tf.convert_to_tensor(test_target, dtype=tf.float32)

        # Combine features and targets into tuples
        # train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_target))
        # test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_target))

        # Add batch dimension to datasets
        # train_dataset = train_dataset.batch(128*2)
        # test_dataset = test_dataset.batch(128*2)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_target)).batch(128*2)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_target)).batch(128*2)


        # Return datasets with input shape and batch dimension
        return (train_dataset, test_dataset, train_features.shape[1:] )
    
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
    model.add(keras.layers.Dense(2, activation='softmax'))
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
generate_2g_csv('dataset.csv', 'dataset.csv')

# load dataset
df = LoadDataSet('dataset.csv')


# remove unwanted columns
df = DatasetPreProcessing(df,['Cell ID', 'Main_Cell ID', 'Distance_km'] )


# angle scalling
df['Azimuth'] = df['Azimuth'].apply(scale_angle)
df['Main_Azimuth'] = df['Main_Azimuth'].apply(scale_angle)


# show dataframe
print(df)

# split data set
(train_data, test_data, input_shape)  = ConverDFToTensorflowDSt('is_NBR', df)

# # implement data normalization
# train_data, test_data = dfDataNormalization(train_data,test_data)

# print(train_data)

# make Machine Learing model
# loaded_model = BuildModle(input_shape) # to rebuild model 

# Load the saved model
loaded_model = tf.keras.models.load_model("models/DNP_2G_ENCODED.keras") # load saved model


# # train model
# loaded_model.fit(train_data, epochs=1400, validation_data=test_data)

# # save the model
# loaded_model.save("models/DNP_2G_ENCODED.keras")


test_input_1 = [32.5394,15.66323,120,32.5352,15.6679,140] # 0

test_input_2 = [32.48817,15.71827,25,32.49186,15.72213,130] # 1

test_input_3 = [32.56781,15.66145,305-360,32.5746,15.65402,240-360] # 0 

test_input_4 = [32.48817,15.71827,230-360,32.4792,15.7097,10] # 1

test_input_5 = [32.652542,15.475766,315-360,32.6297,15.4461,0] # 1

test_input_6 = [32.5394,15.66323,270-360,32.54262,15.66544,160] # 0

test_input_7 = [32.48817,15.71827,120,32.4988,15.7228,260-360] # 1

test_input_8 = [32.5819,15.7279,130,32.56692,15.73135,160] # 0

test_input_9 = [32.652542,15.475766,315-360,32.652542,15.475766,210-360] # 1

test_input_10 = [32.5394,15.66323,0,32.5396,15.6564,260-360] # 0


# test_input = [32.53844,15.44687,237,32.5249,15.4633,0]

df = pd.DataFrame(test_input_1)

scaler = StandardScaler()

normalized_features = scaler.fit_transform(df)


test_input_1 = np.array(normalized_features).reshape(1, -1)


# Make predictions
predictions = loaded_model.predict(test_input_1)


print("Prediction:",int(predictions[0][0]),int(predictions[0][1]))