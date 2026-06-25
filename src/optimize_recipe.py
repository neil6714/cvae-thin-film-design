import torch
from torch import nn
import pandas as pd
import numpy as np
import pickle

# Architecture Definition (Must match training)
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

def run_optimization_pipeline(num_test_cases=10000):
    print(f"Generating {num_test_cases} theoretical material targets...")
    
    # 1. Generate massive search space of target properties
    target_gpc = np.random.uniform(0.5, 3.0, num_test_cases)
    target_ri = np.random.uniform(1.9, 2.1, num_test_cases)
    target_stress = np.random.uniform(-300, 100, num_test_cases)
    
    condition_data = np.column_stack((target_gpc, target_ri, target_stress))

    # 2. Load Deep Learning infrastructure
    with open('models/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    model = CVAE()
    model.load_state_dict(torch.load('models/cvae_weights.pth', weights_only=True))
    model.eval()

    # 3. Scale and Vectorized Decode
    c_scaled = scalers['scaler_c'].transform(condition_data)
    c_tensor = torch.FloatTensor(c_scaled)
    z = torch.randn(num_test_cases, 2)
    
    with torch.no_grad():
        generated_scaled = model.decode(z, c_tensor)
        
    recipes = scalers['scaler_x'].inverse_transform(generated_scaled.numpy())

    # 4. Compile Results into DataFrame
    df = pd.DataFrame({
        'Req_Target_GPC': target_gpc,
        'Req_Target_RI': target_ri,
        'Req_Target_Stress': target_stress,
        'Gen_Temp_C': recipes[:, 0],
        'Gen_Pulse_ms': recipes[:, 1],
        'Gen_Power_W': recipes[:, 2]
    })

    # 5. FITNESS FUNCTION: Define what "Optimal" means
    # Goal: Maximize GPC (weight 10), Minimize absolute stress (weight -0.1), Maximize RI (weight 5)
    df['Fitness_Score'] = (df['Req_Target_GPC'] * 10) - (df['Req_Target_Stress'].abs() * 0.1) + (df['Req_Target_RI'] * 5)

    # 6. Extract the top 5 optimal recipes
    top_candidates = df.sort_values(by='Fitness_Score', ascending=False).head(5)
    
    output_path = 'data/top_optimized_recipes.csv'
    top_candidates.to_csv(output_path, index=False)
    
    print("\n--- TOP 3 OPTIMAL RECIPES FOUND ---")
    for i, (_, row) in enumerate(top_candidates.head(3).iterrows(), 1):
        print(f"\nRank #{i} | Score: {row['Fitness_Score']:.2f}")
        print(f"Targets  -> GPC: {row['Req_Target_GPC']:.2f} | RI: {row['Req_Target_RI']:.2f} | Stress: {row['Req_Target_Stress']:.0f} MPa")
        print(f"Recipe   -> Temp: {row['Gen_Temp_C']:.1f}°C | Pulse: {row['Gen_Pulse_ms']:.1f}ms | Power: {row['Gen_Power_W']:.1f}W")
    
    print(f"\nFull top 5 dataset saved to: {output_path}")

if __name__ == "__main__":
    run_optimization_pipeline()