from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from CLASSIFY import run_model_classification
from CLASSIFY import Get_Models

app = FastAPI()

app.add_middleware( #allows fine tuning of who can call the apis, what methods are allowed etc 
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Specific origins that are allowed to call the api. 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


#creates a class, that expects a model choice (NBE,NBSMS, GRU etc) and then the users message. 
class UserInput(BaseModel):
    ModelChoice: str
    Message: str
    
@app.get("/CLASSIFY/GETModels") #returns all models used in this project using a get method
def APIGETMODELS() -> dict :
    return Get_Models()


#Creates an API endpoint allowing for interaction when the api is "started" (python -m uvicorn TestAPI:app --reload)
@app.post("/CLASSIFY/Detection")
def API_CALL(input_data: UserInput):
    
    try: 
        result = run_model_classification(input_data.Message,input_data.ModelChoice )
        if result is None: 
            return {"error" : "invalid Model Option"}
        if isinstance(result, dict): #checks if the result is a dictionary value (handles the newly added multiple call function)
            return{
                "type": "Multiple",
                "results": result
            }
        elif isinstance(result, tuple) and len(result) == 2: 
            label, confidence = result  # e.g label = "Spam", confidence = 0.95 The errors with GRU and SVM were with the fact it wasn't able to read the array correctly, so i have split the tuple
            return {
                "type": "single",
                "Prediction": str(label),      # e.g "Spam" / "Safe" 
                "Confidence": float(confidence) if confidence is not None else None # e.g 0.95 and checks if the return value is NONE (as SVM returns none, instead of a confidence)
            }
        else: 
            return {"error": "Unexpected result format, call a programmer"}  
    except Exception as e : #exception handling and debugging, keep it here if it breaks before we hand it in, it makes it easier to debug
        import traceback
        print(f"Error in API_CALL: {e}") #fully ripped this error handling from stackoverflow, but it works
        print(traceback.format_exc())
        return {"error": str(e)}

