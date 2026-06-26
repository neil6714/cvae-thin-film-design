import torch
import pandas as pd
import numpy as np
import pickle
import sys
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Add src to path so we can import your custom architectures
sys.path.append('src')
from optimize_recipe import CVAE
from train_ddpm import NoisePredictor, TabularDiffusion

def run_benchmark():
    # Train the ALD Digital Twin
    print("--- 1. Training the ALD Digital Twin ---")
    df = pd.read_csv('data/recipes_and_properties.csv')
    X_oracle = df[['Temperature_C', 'Pulse_Time_ms', 'Plasma_Power_W']]
    y_oracle = df[['Target_GPC', 'Target_Refractive_Index', 'Target_Film_Stress_MPa']]
    
    oracle = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    oracle.fit(X_oracle, y_oracle)
    print(f"Internal Accuracy (R^2): {oracle.score(X_oracle, y_oracle):.4f}\n")

    # Setup 100 Hard Test Targets
    print("--- 2. Setting up 100 Test Targets ---")
    targets = np.random.uniform(low=[0.5, 1.9, -300], high=[3.0, 2.1, 100], size=(100, 3))
    
    with open('models/ddpm_scalers.pkl', 'rb') as f:
        ddpm_scalers = pickle.load(f)
    with open('models/scalers.pkl', 'rb') as f:
        cvae_scalers = pickle.load(f)
        
    c_tensor_cvae = torch.FloatTensor(cvae_scalers['scaler_c'].transform(targets))
    c_tensor_ddpm = torch.FloatTensor(ddpm_scalers['scaler_c'].transform(targets))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load & Run CVAE
    print("--- 3. Running CVAE Generation ---")
    cvae = CVAE()
    cvae.load_state_dict(torch.load('models/cvae_weights.pth', map_location=torch.device('cpu'), weights_only=True))
    cvae.eval()
    
    start_time = time.time()
    with torch.no_grad():
        z = torch.randn(100, 2)
        cvae_scaled = cvae.decode(z, c_tensor_cvae)
    cvae_recipes = cvae_scalers['scaler_x'].inverse_transform(cvae_scaled.numpy())
    cvae_time = time.time() - start_time

    # Load & Run DDPM
    print("--- 4. Running DDPM Generation ---")
    predictor = NoisePredictor().to(device)
    
    state_dict = torch.load('models/ddpm_weights.pth', map_location=device, weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    predictor.load_state_dict(state_dict)
    
    diffusion = TabularDiffusion(predictor, timesteps=100).to(device)
    diffusion.eval()
    
    start_time = time.time()
    c_tensor_ddpm = c_tensor_ddpm.to(device)
    with torch.no_grad():
        ddpm_scaled = diffusion.sample(c_tensor_ddpm, shape=(100, 3))
    ddpm_recipes = ddpm_scalers['scaler_x'].inverse_transform(ddpm_scaled.cpu().numpy())
    ddpm_time = time.time() - start_time

    # Evaluation
    print("\n--- EVALUATION RESULTS ---")
    cvae_predicted_props = oracle.predict(cvae_recipes)
    ddpm_predicted_props = oracle.predict(ddpm_recipes)
    
    # Calculate Mean Absolute Error (Lower is better)
    cvae_mae = mean_absolute_error(targets, cvae_predicted_props)
    ddpm_mae = mean_absolute_error(targets, ddpm_predicted_props)
    
    print(f"CVAE | Inference Time: {cvae_time:.4f}s | Mean Absolute Error: {cvae_mae:.4f}")
    print(f"DDPM | Inference Time: {ddpm_time:.4f}s | Mean Absolute Error: {ddpm_mae:.4f}")
    
    if ddpm_mae < cvae_mae:
        print("\nWinner: DDPM is more accurate")
    else:
        print("\nWinner: CVAE is more accurate")

if __name__ == "__main__":
    run_benchmark()