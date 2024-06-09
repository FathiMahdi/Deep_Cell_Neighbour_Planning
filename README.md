# Deep Cell Neighbour Planning

## description

This project is about using AI for telecommunications network cellular tower planing, the AI model takes the main cel information and the neighbour cell information and predict the relation. The model is based on logistic regression so the output neuron gives 1/0.

## Required parameters

- Main_Longitude
- Main_Latitude
- Main_Azimuth
- Longitude
- Latitude
- Azimuth


## Models

| **Model  Name** | **Description** | **Accuracy** |
| :---            | :---            | :---         |
| decision_tree_model.pkl        | trained with 2G data only | 94.38 %      |
| knn_model.pkl   | trained with 2G data only | 94.16 % |

## How to perform prediction

- check test.py

```python
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


# for single list
single_sample = [32.652542,15.475766,315,32.63956,15.50159,240] # should give 1

# Inference using loaded models
prediction = dnp.predict(multi_list)

print("Prediction: ", prediction)
```
