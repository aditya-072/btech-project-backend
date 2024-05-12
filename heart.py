# ML
import array
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# fast api
from fastapi import FastAPI
from pydantic import BaseModel

from medicine_recommendation.medicine_recommendation import get_medicine_recommendation


# parkinson's input for model prediction
class Heart_input(BaseModel):
    i1: float | None = 0.0
    i2: float | None = 0.0
    i3: float | None = 0.0
    i4: float | None = 0.0
    i5: float | None = 0.0
    i6: float | None = 0.0
    i7: float | None = 0.0
    i8: float | None = 0.0
    i9: float | None = 0.0
    i10: float | None = 0.0
    i11: float | None = 0.0


with open("heart.pkl", "rb") as file:
    loaded_classifier = pickle.load(file)


input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)


def predict_heart(input_data: array = [49, 0, 0, 160, 180, 0, 0, 156, 0, 1, 0]):
    # input_data = (49,"F","NAP",160,180,0,"Normal",156,"N",1,"Flat",1)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_classifier.predict(input_data_reshaped)
    print(prediction)

    ranked_drugs = get_medicine_recommendation("heart")

    ranked_drugs_arr = []
    for index in ranked_drugs["urlDrugName"].keys():
        print(ranked_drugs["urlDrugName"][index])
        ranked_drugs_arr.append(ranked_drugs["urlDrugName"][index])

    if prediction[0] == 0:
        print("The person is fine")
        return {"status": False, "drugs": ranked_drugs_arr}

    else:
        print("The person have heart disease.")
        return {"status": True, "drugs": ranked_drugs_arr}
