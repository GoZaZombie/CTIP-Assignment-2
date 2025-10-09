from fastapi import FastAPI

from pydantic import BaseModel
import CLASSIFY

#Variable app is created and assigned the FastAPI function (we can change the app name, but remember to change commands like @app.post to the new name)
app = FastAPI()


#creates a class, that expects a model choice (NBE,NBSMS, GRU etc) and then the users message. 
class UserInput(BaseModel):
    ModelChoice: str
    Message: str
    

#Creates an API endpoint allowing for interaction when the api is "started" (python -m uvicorn TestAPI:app --reload)
@app.post("/CLASSIFY")
def Classify_SMS_NB(input_data: UserInput):
    result = Classify_SMS_NB(input_data.Message)
    return {"prediction": result}
    