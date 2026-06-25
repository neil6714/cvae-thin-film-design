import numpy as np
import pandas as pd
import os

def generate_ald_dataset(num_samples=10000):
    print("Initializing PEALD kinetic simulation...")
    
    np.random.seed(42)
    
    # 1. PROCESS PARAMETERS (Inputs)
    temp_c = np.random.uniform(150, 350, num_samples)
    pulse_time = np.random.uniform(10, 100, num_samples)
    power = np.random.uniform(50, 500, num_samples)
    
    # 2. MATERIAL PROPERTIES (Targets)
    # Growth Per Cycle (GPC)
    saturation = 1 - np.exp(-pulse_time / 20.0)
    thermal_effect = np.exp(-((temp_c - 250)**2) / 2000) + (0.1 * np.exp(temp_c / 100))
    gpc = 1.2 * saturation * thermal_effect 
    gpc += np.random.normal(0, 0.05, num_samples) 
    gpc = np.clip(gpc, 0.1, 5.0) 
    
    # Refractive Index
    ri = 1.9 + (temp_c / 1000) + (power / 5000)
    ri += np.random.normal(0, 0.02, num_samples)
    
    # Film Stress (MPa)
    stress = -500 + (power * 1.5) - (temp_c * 0.5)
    stress += np.random.normal(0, 20, num_samples)
    
    # 3. COMPILE AND SAVE
    df = pd.DataFrame({
        'Temperature_C': temp_c,
        'Pulse_Time_ms': pulse_time,
        'Plasma_Power_W': power,
        'Target_GPC': gpc,
        'Target_Refractive_Index': ri,
        'Target_Film_Stress_MPa': stress
    })
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/recipes_and_properties.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Success! Generated {num_samples} process runs.")
    print(f"Dataset securely saved to: {file_path}")

if __name__ == "__main__":
    generate_ald_dataset()