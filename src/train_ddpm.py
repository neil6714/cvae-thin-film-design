import torch
from torch import nn
import numpy as np
import pandas as pd
import pickle
import math
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(os.cpu_count())

#  Time Embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Noise Predictor Network 
class NoisePredictor(nn.Module):
    def __init__(self, recipe_dim=3, condition_dim=3, time_emb_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Total inputs: Noisy Recipe + Target Condition + Time Vector
        input_dim = recipe_dim + condition_dim + time_emb_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, recipe_dim) # Outputs predicted noise
        )

    def forward(self, x, time, c):
        t_emb = self.time_mlp(time)
        x_input = torch.cat([x, c, t_emb], dim=1)
        return self.net(x_input)

#The Diffusion Process (Forward & Reverse)
class TabularDiffusion(nn.Module):
    def __init__(self, model, timesteps=100):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Linear variance schedule
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Pre-calculate alpha values for forward noising mechanics
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Forward pass: Jump directly to timestep t and add noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(self, x_start, t, condition, noise=None):
        """Calculate MSE between true added noise and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, condition)
        
        return nn.functional.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, condition, shape):
        """Reverse pass: Iteratively denoise from pure noise to final recipe."""
        device = condition.device
        b = condition.shape[0]
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t, condition)
            
            alpha = self.alphas[t].view(-1, 1)
            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1)
            beta = self.betas[t].view(-1, 1)
            
            # Inject slight stochasticity during reverse steps (except final step)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
        return x

# Data Loading and Training Loop
def train_ddpm(data_path='data/recipes_and_properties.csv', epochs=500, batch_size=128):
    print("Loading dataset for Diffusion Model...")
    df = pd.read_csv(data_path)
    
    # Split features (X) and conditions (C)
    recipes = df[['Temperature_C', 'Pulse_Time_ms', 'Plasma_Power_W']].values
    conditions = df[['Target_GPC', 'Target_Refractive_Index', 'Target_Film_Stress_MPa']].values
    
    # Scale data 
    scaler_x = StandardScaler()
    scaler_c = StandardScaler()
    
    recipes_scaled = scaler_x.fit_transform(recipes)
    conditions_scaled = scaler_c.fit_transform(conditions)
    
    # Save scalers for the inference script later
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/ddpm_scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_x': scaler_x, 'scaler_c': scaler_c}, f)
        
    # Create PyTorch DataLoaders
    dataset = TensorDataset(torch.FloatTensor(recipes_scaled), torch.FloatTensor(conditions_scaled))
    optimal_cores = max(1, os.cpu_count() // 2) 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=optimal_cores, 
        pin_memory=True)
    
    # Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    predictor = NoisePredictor().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Parallelizing across {torch.cuda.device_count()} GPUs...")
        predictor = nn.DataParallel(predictor)
    diffusion = TabularDiffusion(predictor, timesteps=100).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    # PyTorch Training Loop
    predictor.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_c in dataloader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)
            
            # Randomly sample a timestep 't' for every recipe in the batch
            t = torch.randint(0, diffusion.timesteps, (batch_x.shape[0],), device=device).long()
            
            optimizer.zero_grad()
            loss = diffusion.p_loss(batch_x, t, batch_c) # Calculate MSE Noise Loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Average Loss: {total_loss/len(dataloader):.4f}")
            
    # Save the trained weights
    torch.save(predictor.state_dict(), 'models/ddpm_weights.pth')
    print("\nTraining complete. Weights saved to models/ddpm_weights.pth")

if __name__ == "__main__":
    train_ddpm()