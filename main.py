# Fast Api
import uvicorn
from typing import Annotated
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medicine_recommendation.medicine_recommendation import get_medicine_recommendation

from pydantic import BaseModel, Field

# diabetes
from diabetes import predict_diabetes, Dia_input
from heart import predict_heart, Heart_input
from parkinsons import predict_parkinsons, Par_input


app = FastAPI()

# Allow all origins (this disables CORS protection)
# Replace this with your specific requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # You can replace "*" with specific HTTP methods
    allow_headers=["*"],  # You can replace "*" with specific headers
)


@app.post("/heart")
async def heart(data: dict):
    # arr = [
    #     Heart_input.i1,
    #     Heart_input.i2,
    #     Heart_input.i3,
    #     Heart_input.i4,
    #     Heart_input.i5,
    #     Heart_input.i6,
    #     Heart_input.i7,
    #     Heart_input.i8,
    #     Heart_input.i9,
    #     Heart_input.i10,
    #     Heart_input.i11,
    # ]
    # return predict_heart(arr)
    # return get_medicine_recommendation()
    return predict_heart(data["data"])
    return data


@app.post("/diabetes")
async def diabetes(data: dict):
    # print(data.data)
    return predict_diabetes(data.data)
    # arr = [
    #     Dia_input.i1,
    #     Dia_input.i2,
    #     Dia_input.i3,
    #     Dia_input.i4,
    #     Dia_input.i5,
    #     Dia_input.i6,
    #     Dia_input.i7,
    #     Dia_input.i8,
    # ]
    # return predict_diabetes(arr)
    # return data
    return {"ldjfls"}

    # return data


@app.post("/parkinsons")
async def parkinsons(data: dict):

    # arr = [
    #     Par_input.i1,
    #     Par_input.i2,
    #     Par_input.i3,
    #     Par_input.i4,
    #     Par_input.i5,
    #     Par_input.i6,
    #     Par_input.i7,
    #     Par_input.i8,
    #     Par_input.i9,
    #     Par_input.i10,
    #     Par_input.i11,
    #     Par_input.i12,
    #     Par_input.i13,
    #     Par_input.i14,
    #     Par_input.i15,
    #     Par_input.i16,
    #     Par_input.i17,
    #     Par_input.i18,
    #     Par_input.i19,
    #     Par_input.i20,
    #     Par_input.i22,
    #     Par_input.i21,
    # ]
    
    return predict_parkinsons(data.data)
    return data


@app.post("/alzheimer")
async def alzheimer(data: dict):
    return {"Alzheimer not found."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
