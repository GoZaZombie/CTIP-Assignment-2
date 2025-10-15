from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from CLASSIFY import Classify_SMS_NB
from CLASSIFY import Classify_EMAIL_NB
from CLASSIFY import run_model_classification
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # You can use ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],            # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],            # Allow all headers
)




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

