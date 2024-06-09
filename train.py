from DNP import DNP
from sklearn.model_selection import train_test_split
from csv_edit import process_data_set
dnp = DNP()

# Load data
# data = dnp.load_data('dataset_with_cos.csv')
process_data_set()

data = dnp.load_data('GUL_DS.csv')

# Prepare data
# X = data[['Main_Longitude', 'Main_Latitude', 'Main_Azimuth','Longitude', 'Latitude', 'Azimuth',"cos_Main_Azimuth","cos_Azimuth"]]
X = data[['Main_Longitude', 'Main_Latitude', 'Main_Azimuth','Longitude', 'Latitude', 'Azimuth','Distance_km']]
y = data['is_NBR']

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)



# Train models (knn and decision tree)
dnp.train_models(X_train, y_train, X_test, y_test,r=123)

# or train with tuned prameters change x and y default(n=1, r=42)

# dnp.train_models(X_train, y_train, X_test, y_test,n=1,r=42)


# Save trained models default in models folder
dnp.save_models()

# or you can specify the models location
# where x/x, y/y, and z/z are the directories

# dnp.save_models(knn_path='x/x', dt_path='y/y', scaler_path='z/z')