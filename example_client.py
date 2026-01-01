import requests
import numpy as np
import time

# Configuration
API_URL = "http://localhost:8000/predict"

def get_market_data_batch():
    """
    Simulates fetching a batch of market candles (Open, High, Low, Close).
    Here we generate synthetic data for demonstration.
    Shape: (Batch Size=1, Sequence Length=50, Features=1)
    """
    # Generate random walk price data
    price = 100 + np.cumsum(np.random.randn(50))
    # Normalize (Standard Scaling is recommended)
    price = (price - np.mean(price)) / np.std(price)
    
    # Reshape for model: [Batch, Time, Feat]
    return price.reshape(1, 50, 1).tolist()

def analyze_market():
    print("🔌 Connecting to Inversion Transformer Engine...")
    
    # 1. Prepare Data
    data = get_market_data_batch()
    
    # 2. Send Request
    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json={"inputs": data, "return_repr": False}
        )
        latency = (time.time() - start_time) * 1000
        
        # 3. Handle Response
        if response.status_code == 200:
            result = response.json()
            prediction = result["predictions"][0][-1][0] # Last step prediction
            print(f"✅ Success ({latency:.1f}ms)")
            print(f"📈 Predicted Next Value: {prediction:.4f}")
            
            # Simple Trading Logic Example
            last_price = data[0][-1][0]
            if prediction > last_price:
                print("💡 Signal: BUY (Predicted Uptrend)")
            else:
                print("💡 Signal: SELL (Predicted Downtrend)")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect. Is the Docker container running?")

if __name__ == "__main__":
    analyze_market()
