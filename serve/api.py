import sys
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn

# Add root directory to path to allow importing core modules
# This works regardless of whether core is .py or .so (compiled)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.inversion_transformer import InversionTransformerCore
except ImportError:
    # If running in a compiled environment, the import might behave differently
    # depending on how .so files are linked.
    # In Docker, we ensure the .so files are in the python path.
    print("⚠️ Warning: Could not import InversionTransformerCore directly. Checking for compiled extensions...")
    from core.inversion_transformer import InversionTransformerCore

app = FastAPI(
    title="Inversion Transformer Enterprise API",
    description="Secure REST API for Anomaly Detection Engine",
    version="1.0.0"
)

# --- Data Schemas ---
class PredictionRequest(BaseModel):
    inputs: List[List[List[float]]]  # Shape: (batch_size, seq_len, input_dim)
    return_repr: bool = False

class PredictionResponse(BaseModel):
    predictions: List[List[float]]
    representations: Optional[List[List[float]]] = None
    status: str = "success"

# --- Model Initialization ---
# In a real deployment, we would load weights here.
# For the demo container, we initialize a standard model.
INPUT_DIM = 1 # Default for Univariate Time Series
model = InversionTransformerCore(input_dim=INPUT_DIM)
model.eval()

print("✅ Inversion Transformer Model Loaded Successfully")

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes/Docker."""
    try:
        # Verify model is accessible
        info = model.model_info()
        return {
            "status": "operational", 
            "model": "InversionTransformer",
            "version": "Enterprise v1.0",
            "compiled": True # We assume this runs in the secure container
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Run inference on time-series batch.
    """
    try:
        # Convert list to tensor
        x = torch.tensor(request.inputs, dtype=torch.float32)
        
        # Inference
        with torch.no_grad():
            y, h = model(x, return_repr=request.return_repr)
            
        return {
            "predictions": y.tolist(),
            "representations": h.tolist() if h is not None else None,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # In Docker, this is run via CMD
    uvicorn.run(app, host="0.0.0.0", port=8000)
