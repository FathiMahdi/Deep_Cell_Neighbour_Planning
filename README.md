# Deep_Cell_Neighbour_Planning

## Required parameters

- Main_Longitude
- Main_Latitude
- Main_Azimuth
- Longitude
- Latitude
- Azimuth
- Distance_km


## How to perform prediction

```python
loaded_model = tf.keras.models.load_model("models/DNP_2G.h5") # load saved model

test_input = [32.5399,15.5947,315,32.5399,15.5947,125,0]

test_input = np.array(test_input)

test_input = test_input.reshape(1, -1)

predictions = loaded_model.predict(test_input)

print("Prediction:", predictions)
```
