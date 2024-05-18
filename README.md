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
- Distance_km

## Models

| **Model  Name** | **Description** | **Accuracy** |
| :---            | :---            | :---         |
| DNP_2G          | trained with 2G data only | 93.4 %      |
| DNP_ALL         | trained with 2G, 3G, and 4G data | 86.3 % |

## How to perform prediction

```python
loaded_model = tf.keras.models.load_model("models/DNP_2G.h5") # load saved model

test_input = [32.5399,15.5947,315,32.5399,15.5947,125,0]

test_input = np.array(test_input)

test_input = test_input.reshape(1, -1)

predictions = loaded_model.predict(test_input)

print("Prediction:", predictions)
```
