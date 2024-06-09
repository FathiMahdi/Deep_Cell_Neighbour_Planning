from DNP import DNP
from sklearn.model_selection import train_test_split

dnp = DNP()

# Load data
data = dnp.load_data('dataset.csv')

# Load models
dnp.load_model("models/decision_tree_model.pkl","models/scaler.pkl") # you also can specify you file directory and name


###########################

# [!NOTE]
# No need to modify the angle 
# Just use the same azimuth angle representation (! same as in your dataset)

########################


# for multi data like filtered data
multi_list = \
[
    [32.5711,15.4985,205,32.57133,15.50351,285],
    [32.652542,15.475766,315,32.63956,15.50159,240],
    [32.71825,15.68753,270,32.71825,15.68753,180],
    [32.5314,15.6435,180,32.52401,15.63674,110],
    [32.5314,15.6435,180,32.54,15.6419,270],
    [32.5314,15.6435,180,32.538704,15.63589,260],
    [32.5711,15.4985,205,32.56532,15.4994,0],
    [32.4365,15.6193,240,32.4321,15.6292,290],
    [32.4365,15.6193,240,32.4321,15.6292,120],
    [32.4365,15.6193,240,32.4321,15.6292,0],
]

new_sample = [
[32.5473,15.5691,20,32.55084,15.571843,300,500] ,
[32.5473,15.5691,20,32.5444,15.5741,130,500]    ,
[32.5473,15.5691,20,32.5399,15.5735,110,500]    ,
[32.5473,15.5691,20,32.5399,15.5735,240,500]    ,
[32.5473,15.5691,20,32.5399,15.5735,0,500]  ,
[32.5473,15.5691,20,32.5502,15.577,180,500] ,
[32.5473,15.5691,20,32.5436,15.564,160,500] ,
[32.5473,15.5691,20,32.5431,15.5692,180,500]    ,
[32.5473,15.5691,20,32.5431,15.5692,300,500]    ,
[32.5473,15.5691,20,32.5459,15.5784,140,500]    ,
[32.5473,15.5691,20,32.559681,15.575294,70,500] ,
[32.5473,15.5691,20,32.5562,15.56594,310,500]   ,
[32.5473,15.5691,20,32.55084,15.571843,190,500]
]


new_sample_with_distance = \
[[32.47679,15.69704,255,32.4632,15.7041,275,1.652],
[32.47679,15.69704,255,32.4632,15.7041,350,1.652],
[32.47679,15.69704,255,32.4903,15.6954,240,1.458],
[32.47679,15.69704,255,32.4829,15.6845,110,1.533],
[32.47679,15.69704,255,32.4829,15.6845,0,1.533],
[32.46727,15.68789,345,32.46727,15.68789,120,0],
[32.46727,15.68789,345,32.46727,15.68789,250,0],
[32.46727,15.68789,345,32.4705,15.6942,245,0.779],
[32.46727,15.68789,345,32.45993,15.69589,130,1.183],
[32.46727,15.68789,345,32.46785,15.69673,238,0.979],
[32.46727,15.68789,345,32.45579,15.68708,40,1.233],
[32.46727,15.68789,345,32.47527,15.68948,6,0.875],
[32.46727,15.68789,345,32.46785,15.69673,108,0.979],
[32.46727,15.68789,345,32.47679,15.69704,255,1.437],
[32.46727,15.68789,345,32.45579,15.68708,340,1.233],
[32.46727,15.68789,345,32.47527,15.68948,240,0.875]]

# for single list
single_sample = [32.652542,15.475766,315,32.63956,15.50159,240] # should give 1

# Inference using loaded models
prediction = dnp.predict(new_sample)

print("Prediction: ", prediction)