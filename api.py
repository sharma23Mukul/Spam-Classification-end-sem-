from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import preprocess

# Define the FastAPI application
app = FastAPI(
    title="Probabilistic Spam Classifier API",
    description="A stateless FastAPI wrapper for our Naïve Bayes Spam Classifier.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
# This is REQUIRED for the Gmail Chrome Extension to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (can be restricted to extension ID)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ─── Load Model on Startup ─────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_model.pkl")
loaded_model = None

@app.on_event("startup")
def load_saved_model():
    """
    Load the serialized (pickled) Multinomial Naïve Bayes model into memory 
    when the FastAPI server starts. This allows predictions to happen in 
    milliseconds since we bypass training.
    """
    global loaded_model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Please run 'python save_model.py' first to train and save the model."
        )
    
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
        loaded_model = bundle["model"]
        print(f"✅ Model loaded successfully from {MODEL_PATH}!")
        print(f"   Vocabulary Size: {bundle['vocabulary_size']}")
        print(f"   Messages trained on: {bundle['total_messages_trained']}")


# ─── API Endpoints ─────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_confident: bool
    explanation: str
    probabilities: dict

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "online", "model": "Multinomial Naïve Bayes"}

@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: PredictionRequest):
    """
    Predict whether a given message is spam or ham.
    This endpoint is designed to be called by external applications, 
    like the Gmail Chrome Extension.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    global loaded_model
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model failed to load. Please restart server.")

    # 1. Preprocess the raw input text exactly as we did during training
    tokens = preprocess(request.message)

    if not tokens:
        # If the message only contained stop-words or empty strings
        return PredictionResponse(
            prediction="uncertain",
            confidence=0.0,
            is_confident=False,
            explanation="No meaningful words remaining after preprocessing.",
            probabilities={"spam": 0.5, "ham": 0.5}
        )

    # 2. Feed the tokens into our custom Naïve Bayes model
    # We use a 70% confidence threshold to handle ambiguous/tricky texts
    result = loaded_model.predict_with_confidence(tokens, confidence_threshold=0.70)

    # 3. Return the JSON response
    return PredictionResponse(
        prediction=result['prediction'],
        confidence=result['confidence'],
        is_confident=result['is_confident'],
        explanation=result['explanation'],
        probabilities=result['probabilities']
    )

if __name__ == "__main__":
    import uvicorn
    # If this file is run directly, start the uvicorn server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
