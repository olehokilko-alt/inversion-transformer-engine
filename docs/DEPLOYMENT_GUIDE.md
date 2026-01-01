# Inversion Transformer Deployment Guide

## 🔒 Security Notice
This container is hardened. The core algorithmic logic (`adaptive_controller.py`, `inversion_transformer.py`) has been compiled into binary C-extensions (`.so` files) and the original source code has been removed. This protects the Intellectual Property (IP) while allowing full functionality.

## 🚀 Running the Engine (Docker)

### 1. Build the Secure Image
```bash
docker build -t inversion-transformer-enterprise:v1.0 .
```
*Note: During the build process, you will see `Compiling...` messages. This is the Cython compiler converting Python to C.*

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name isip-engine inversion-transformer-enterprise:v1.0
```

### 3. Verify Deployment
Check the health status:
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "operational",
  "model": "InversionTransformer",
  "version": "Enterprise v1.0",
  "compiled": true
}
```

---

## 🔌 API Reference

### Predict Endpoint
**POST** `/predict`

**Input (JSON):**
```json
{
  "inputs": [[[0.5], [0.6], [0.7], [0.4], [0.2]]],
  "return_repr": false
}
```
*(Shape: Batch Size x Sequence Length x Input Dimension)*

**Output (JSON):**
```json
{
  "predictions": [[0.35]],
  "status": "success"
}
```

---

## 🛠 Integration (Python Client Example)
```python
import requests
import numpy as np

# Prepare data (1 batch, 50 time steps, 1 feature)
data = np.random.randn(1, 50, 1).tolist()

response = requests.post(
    "http://localhost:8000/predict",
    json={"inputs": data}
)

print(response.json())
```
