import torch
from torch import nn
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore') # Suppress sklearn scaling warnings

#  Architecture Definition
class CVAE(nn.Module):
    def __init__(self, recipe_dim=3, condition_dim=3, latent_dim=2):
        super(CVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(recipe_dim + condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, recipe_dim)
        )

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def generate_recipe(target_gpc, target_ri, target_stress):
    # Load scalers
    with open('models/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    scaler_x = scalers['scaler_x']
    scaler_c = scalers['scaler_c']

    # Initialize model and load trained weights
    model = CVAE()
    model.load_state_dict(torch.load('models/cvae_weights.pth', weights_only=True))
    model.eval()

    # Format targets and scale
    targets = np.array([[target_gpc, target_ri, target_stress]])
    c_scaled = scaler_c.transform(targets)
    c_tensor = torch.FloatTensor(c_scaled)

    # Sample random point from latent space (Standard Normal Distribution)
    z = torch.randn(1, 2)

    # Generate recipe via Decoder
    with torch.no_grad():
        generated_scaled = model.decode(z, c_tensor)

    # Inverse transform to retrieve physical units
    recipe = scaler_x.inverse_transform(generated_scaled.numpy())[0]

    # Output formatting
    print("\n" + "="*50)
    print("  DEEP LEARNING INVERSE DESIGN: RECIPE GENERATED  ")
    print("="*50)
    print(f"TARGET PROPERTIES:")
    print(f" - Growth Per Cycle : {target_gpc:.2f} Å/cyc")
    print(f" - Refractive Index : {target_ri:.2f}")
    print(f" - Film Stress      : {target_stress:.2f} MPa")
    print("-" * 50)
    print(f"OPTIMAL ALD RECIPE:")
    print(f" - Temperature  : {recipe[0]:.1f} °C")
    print(f" - Pulse Time   : {recipe[1]:.1f} ms")
    print(f" - Plasma Power : {recipe[2]:.1f} W")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test execution: Requesting a custom film
    generate_recipe(target_gpc=7, target_ri=7, target_stress=-7)