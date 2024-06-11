import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import math
import numpy as np 

# Deep Cell Neighbour Planning
class DNP:
    def __init__(self):

        # No need to expose the ML models this is only for testing (you may keep the user selection one)
        self.knn_model = None
        self.dt_model = None
        self.scaler = None
        self.svm_model = None


        self.user_selected_model = None
    
    
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def convert_to_vector(self,longitude, latitude, azimuth):
        # Convert latitude and longitude to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        
        # Convert azimuth to radians
        az_rad = math.radians(azimuth)
        
        # Calculate x, y, z components of the vector
        x = math.cos(lat_rad) * math.cos(lon_rad) * math.cos(az_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad) * math.cos(az_rad)
        z = math.sin(lat_rad) * math.sin(az_rad)
        
        return x, y, z
    
    def train_models(self, X_train, y_train, X_test, y_test, n=1, r=42):
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train KNN model
        self.knn_model = KNeighborsClassifier(n_neighbors=2)
        self.knn_model.fit(X_train_scaled, y_train)
        knn_pred = self.knn_model.predict(X_test_scaled)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        print("KNN Accuracy:", knn_accuracy)
        
        # Train Decision Tree model
        self.dt_model = DecisionTreeClassifier(criterion="entropy")
        self.dt_model.fit(X_train_scaled, y_train)
        dt_pred = self.dt_model.predict(X_test_scaled)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        print("Decision Tree Accuracy:", dt_accuracy)


    def save_models(self, knn_path='models/knn_model.pkl', dt_path='models/decision_tree_model.pkl', scaler_path='models/scaler.pkl'):
        joblib.dump(self.knn_model, knn_path)
        joblib.dump(self.dt_model, dt_path)
        joblib.dump(self.scaler, scaler_path)


    def predict(self, input_data):

        if isinstance(input_data, pd.DataFrame):
            X = input_data
        
        elif isinstance(input_data[0], list):
  
            X = pd.DataFrame(input_data, columns=['Main_Longitude', 'Main_Latitude', 'Main_Azimuth','Longitude', 'Latitude', 'Azimuth','Distance_km'])

        else:
     
            X = pd.DataFrame([input_data], columns=['Main_Longitude', 'Main_Latitude', 'Main_Azimuth','Longitude', 'Latitude', 'Azimuth','Distance_km'])
        

        X_scaled = self.scaler.transform(X)
        

        prediction = self.user_selected_model.predict(X_scaled)
        confidence = self.user_selected_model.predict_proba(X, check_input=True)

        return prediction, confidence


    def load_model(self, model_path,scaler_path):
        self.user_selected_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)