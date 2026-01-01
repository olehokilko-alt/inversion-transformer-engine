
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import json

# Add current directory to path
sys.path.append(os.getcwd())

from core.adaptive_controller import AdaptiveInversionController

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_report(results, filename="test_report.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def plot_analysis(name, t, data, anomaly_label, output_dir="assets"):
    print(f"🔍 Running Analysis for: {name}...")
    
    controller = AdaptiveInversionController(verbose=False)
    
    seq_len = 50
    inv_weights = []
    plot_t = []
    
    # Simulate streaming data
    for i in range(len(data) - seq_len):
        window = data[i : i + seq_len]
        X_batch = window.reshape(1, seq_len, 1)
        # Dummy target
        Y_batch = np.array([0.0])
        
        weight, _ = controller.recommend(X_batch, Y_batch)
        inv_weights.append(weight)
        plot_t.append(t[i + seq_len - 1])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Signal Plot
    ax1.plot(t, data, 'b-', linewidth=1, alpha=0.8, label='Raw Signal')
    ax1.set_title(f'Case Study: {name} - {anomaly_label}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # AI Confidence Plot
    ax2.plot(plot_t, inv_weights, 'r-', linewidth=2, label='Inversion Weight (AI Attention)')
    
    # Thresholds
    ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Critical Threshold')
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Stable Threshold')
    
    # Highlight detection zone
    ax2.fill_between(plot_t, inv_weights, 0.8, where=np.array(inv_weights)>0.8, color='red', alpha=0.3, label='Anomaly Detected')
    
    ax2.set_ylabel('Inversion Weight (λ)', fontsize=12)
    ax2.set_xlabel('Time / Steps', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    filename = os.path.join(output_dir, f"proof_{name.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✅ Generated proof asset: {filename}")
    
    # Return metrics for report
    avg_weight = np.mean(inv_weights)
    max_weight = np.max(inv_weights)
    detection_speed = "Instant (<50ms)" # Placeholder for simulation
    
    return {
        "case": name,
        "anomaly": anomaly_label,
        "avg_inversion": float(avg_weight),
        "max_inversion": float(max_weight),
        "detection_status": "SUCCESS" if max_weight > 0.8 else "FAILED",
        "proof_image": filename
    }

def run_universal_suite():
    print("🚀 Starting Inversion Transformer Enterprise Validation Suite...")
    ensure_dir("assets")
    
    report_data = []

    # --- CASE 1: FinTech (Flash Crash) ---
    t = np.linspace(0, 200, 400)
    price = 100 + t * 0.5 + np.random.randn(400) * 2 # Trend + Noise
    # Flash Crash event
    price[200:220] -= 30 
    price[220:250] += np.random.randn(30) * 10 # Post-crash volatility
    
    res1 = plot_analysis("FinTech_Crypto_Market", t, price, "Flash Crash & Volatility")
    report_data.append(res1)

    # --- CASE 2: MedTech (Arrhythmia) ---
    t = np.linspace(0, 10, 1000)
    ecg = signal.sawtooth(2 * np.pi * 1.2 * t, 0.5) 
    ecg += 0.1 * np.random.randn(1000)
    # Atrial Fibrillation (Irregular, fast, chaotic)
    ecg[500:700] = np.random.randn(200) * 0.8 + 0.5 * np.sin(2*np.pi*10*t[500:700])
    
    res2 = plot_analysis("MedTech_Cardiac_ECG", t, ecg, "Atrial Fibrillation")
    report_data.append(res2)

    # --- CASE 3: Industrial IoT (Bearing Fault) ---
    t = np.linspace(0, 100, 1000)
    # Harmonics
    vib = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    vib += np.random.randn(1000) * 0.1
    # Developing fault (amplitude modulation + noise increase)
    vib[600:] *= (1 + 0.01 * (np.arange(400))) 
    vib[600:] += np.random.randn(400) * 0.5
    
    res3 = plot_analysis("Industrial_IoT_Motor", t, vib, "Progressive Bearing Wear")
    report_data.append(res3)

    # --- CASE 4: IT Infrastructure (DDoS) ---
    t = np.linspace(0, 24, 1440) # Minutes in a day
    requests = 1000 + 500 * np.sin(2 * np.pi * (t-6)/24) # Daily cycle
    requests = np.maximum(requests, 0)
    requests += np.random.poisson(50, 1440) # Random user traffic
    # DDoS Attack (Sudden massive spike)
    requests[800:860] += 5000 + np.random.randn(60) * 500
    
    res4 = plot_analysis("IT_Server_Traffic", t, requests, "DDoS Attack Pattern")
    report_data.append(res4)

    # Save Summary Report
    save_report(report_data, "assets/validation_summary.json")
    print("\n🏆 Validation Suite Completed. All assets generated.")

if __name__ == "__main__":
    run_universal_suite()
