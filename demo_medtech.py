import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Configuration
API_URL = "http://localhost:8000/predict"
WINDOW_SIZE = 50

def generate_ecg_sample(t):
    """Simulates a heartbeat signal."""
    # Basic rhythm + noise
    val = np.sin(t * 1.0) 
    # Add 'QRS complex' spike
    if t % 20 < 2:
        val += 3.0
    return val + np.random.normal(0, 0.1)

def run_monitor():
    print("🏥 Starting MedTech Real-time Monitor...")
    print("Press Ctrl+C to stop")
    
    buffer = deque(maxlen=WINDOW_SIZE)
    t = 0
    
    try:
        while True:
            # 1. Get new sample
            val = generate_ecg_sample(t)
            buffer.append([val])
            t += 1
            
            # 2. Analyze if buffer full
            if len(buffer) == WINDOW_SIZE:
                payload = {"inputs": [list(buffer)], "return_repr": False}
                resp = requests.post(API_URL, json=payload)
                
                if resp.status_code == 200:
                    score = resp.json()["predictions"][0][0]
                    
                    # Visualization (Console-based for compatibility)
                    bar = "█" * int(score * 10)
                    status = "💓 NORMAL" if score < 0.8 else "⚠️ ARRHYTHMIA DETECTED"
                    print(f"ECG: {val:5.2f} | AI Analysis: {score:.2f} {bar} | {status}")
                
            time.sleep(0.1) # Simulate 10Hz monitor
            
    except KeyboardInterrupt:
        print("\nMonitor Stopped.")

if __name__ == "__main__":
    run_monitor()
