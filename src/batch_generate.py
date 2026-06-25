import torch
from torch import nn
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Architecture Definition 
class CVAE(nn.Module):
    def __init__(self, recipe_dim=3, condition_dim=3, latent_dim=2):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(recipe_dim + condition_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + condition_dim, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, recipe_dim))

    def encode(self, x, c):
        return self.fc_mu(self.encoder(torch.cat([x, c], dim=1))), self.fc_logvar(self.encoder(torch.cat([x, c], dim=1)))

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

def batch_generate(input_csv_path, output_csv_path):
    print(f"Loading target properties from {input_csv_path}...")
    df_targets = pd.read_csv(input_csv_path)
    condition_data = df_targets[['Target_GPC', 'Target_Refractive_Index', 'Target_Film_Stress_MPa']].values

    # Load scalers and model
    with open('models/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    model = CVAE()
    model.load_state_dict(torch.load('models/cvae_weights.pth', weights_only=True))
    model.eval()

    # Scale inputs to match training distributions
    c_scaled = scalers['scaler_c'].transform(condition_data)
    c_tensor = torch.FloatTensor(c_scaled)

    # Vectorized Latent Sampling: Generate Z vectors for ALL rows simultaneously
    num_samples = len(condition_data)
    z = torch.randn(num_samples, 2)
    
    # Batch Decode
    with torch.no_grad():
        generated_scaled = model.decode(z, c_tensor)
        
    # Reverse scaling to restore physical units
    recipes = scalers['scaler_x'].inverse_transform(generated_scaled.numpy())
    
    # Append generated parameters to the original dataframe
    df_output = df_targets.copy()
    df_output['Generated_Temp_C'] = recipes[:, 0].round(1)
    df_output['Generated_Pulse_ms'] = recipes[:, 1].round(1)
    df_output['Generated_Power_W'] = recipes[:, 2].round(1)
    
    df_output.to_csv(output_csv_path, index=False)
    print(f"Success! Generated {num_samples} custom recipes.")
    print(f"Saved to: {output_csv_path}")

if __name__ == "__main__":
    input_path = 'data/batch_targets.csv'
    output_path = 'data/batch_generated_recipes.csv'
    
    # Auto-generate a test file if one does not exist
    if not os.path.exists(input_path):
        pd.DataFrame({
            'Target_GPC': [1.0, 1.2, 1.5, 0.8, 1.3],
            'Target_Refractive_Index': [1.9, 2.0, 2.1, 1.95, 2.05],
            'Target_Film_Stress_MPa': [-100, -50, 0, -200, -150]
        }).to_csv(input_path, index=False)
        
    batch_generate(input_path, output_path)