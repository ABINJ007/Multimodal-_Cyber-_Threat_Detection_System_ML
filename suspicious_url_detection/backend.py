from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Request model
class URLRequest(BaseModel):
    url: str

# Dummy values for mse and threshold — replace with your actual model's results
RECONSTRUCTION_THRESHOLD = 0.05

@app.post("/scan")
async def scan_url(data: URLRequest):
    url = data.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # --- Dummy model inference logic ---
    # Replace this with your autoencoder model prediction and mse calculation
    
    # Example: Let's pretend we compute an MSE for this URL
    mse = np.random.uniform(0, 0.1)  # dummy mse for demonstration
    
    # If mse > threshold, treat as suspicious/fake URL
    threat_detected = mse > RECONSTRUCTION_THRESHOLD
    message = "Fake URL" if threat_detected else "Safe URL"
    
    return {
        "threat": threat_detected,
        "message": message,
        "mse": mse,
        "threshold": RECONSTRUCTION_THRESHOLD
    }
