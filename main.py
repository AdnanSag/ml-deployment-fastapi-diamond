from fastapi import FastAPI,Request,Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import pandas as pd
"""
#testing pickle
def main():
    with open ("diamond_model_complete.pkl","rb")as f:
        saved_data=pickle.load(f)
    model=saved_data["model"]

    X_test=pd.read_csv("testdata.csv")
    print(model.predict(X_test))

if __name__ == "__main__":
    main()

"""

app=FastAPI()

templates=Jinja2Templates(directory="templates")


with open ("diamond_model_complete.pkl","rb")as f:
    saved_data=pickle.load(f)
    model=saved_data["model"]
    encoders=saved_data["encoders"]
    scaler=saved_data["scaler"]



class DiamondFeatures(BaseModel):
    carat : float
    cut : str
    color : str
    clarity : str
    depth : float
    table : float
    x : float
    y: float
    z: float

@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})
    
@app.post("/predict")
async def predict(features: DiamondFeatures):
    input_data=pd.DataFrame([features.model_dump()])
    for col in['cut','color','clarity']:
        input_data[col] = encoders[col].transform(input_data[col])
    input_scaled =scaler.transform(input_data)
    prediction=model.predict(input_scaled)
    return {"predicted_price":prediction[0]}
