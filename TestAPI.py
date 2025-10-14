from fastapi import FastAPI

from pydantic import BaseModel
from CLASSIFY import Classify_SMS_NB
from CLASSIFY import Classify_EMAIL_NB
from CLASSIFY import run_model_classification

#Variable app is created and assigned the FastAPI function (we can change the app name, but remember to change commands like @app.post to the new name)
app = FastAPI()


#creates a class, that expects a model choice (NBE,NBSMS, GRU etc) and then the users message. 
class UserInput(BaseModel):
    ModelChoice: str
    Message: str
    



#Creates an API endpoint allowing for interaction when the api is "started" (python -m uvicorn TestAPI:app --reload)
@app.post("/CLASSIFY/Detection")
def API_CALL(input_data: UserInput):
    result = run_model_classification(input_data.Message,input_data.ModelChoice )
    print(f"Gets to the Return function")
    return {"Prediction": result}

