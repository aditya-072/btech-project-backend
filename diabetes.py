# ML
import array
import pickle
import numpy as np

# fast api
from fastapi import FastAPI
from pydantic import BaseModel


class Dia_input(BaseModel):
    i1: float | None = 0.0
    i2: float | None = 0.0
    i3: float | None = 0.0
    i4: float | None = 0.0
    i5: float | None = 0.0
    i6: float | None = 0.0
    i7: float | None = 0.0
    i8: float | None = 0.0


with open("random_forest_model.pkl", "rb") as file:
    loaded_classifier = pickle.load(file)


input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)


def predict_diabetes(input_data: array):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_classifier.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        # print("The person is not diabetic")
        return {"The person is not diabetic"}
    else:
        # print("The person is diabetic")
        return {"The person is diabetic"}
