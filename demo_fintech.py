import sys
import os
import requests
import time
import pandas as pd
import numpy as np
from adapters.csv_adapter import CSVAdapter

# Configuration
API_URL = "http://localhost:8000/predict"
CSV_FILE = "finance_history.csv"

def generate_dummy_csv():
    """Generates a dummy CSV for testing if not exists."""
    print("Generating dummy financial data...")
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="1min"),
        "close": np.cumsum(np.random.randn(1000)) + 100
    })
    df.to_csv(CSV_FILE, index=False)

def run_backtest():
    if not os.path.exists(CSV_FILE):
        generate_dummy_csv()
        
    print(f"🚀 Starting FinTech Backtest on {CSV_FILE}...")
    adapter = CSVAdapter(CSV_FILE)
    
    signals = []
    
    # Process batch by batch
    for batch in adapter.load_batches(batch_size=50, value_column="close"):
        try:
            # Prepare payload for API
            payload = {
                "inputs": [ [[x] for x in batch] ], # Shape: (1, 50, 1)
                "return_repr": False
            }
            
            resp = requests.post(API_URL, json=payload)
            if resp.status_code == 200:
                pred = resp.json()["predictions"][0][0]
                signals.append(pred)
                print(f"✅ Processed Batch. Pred: {pred:.4f}")
            else:
                print(f"❌ Error: {resp.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
            break
            
    print(f"\n🏆 Backtest Complete. Processed {len(signals)} steps.")

if __name__ == "__main__":
    run_backtest()
