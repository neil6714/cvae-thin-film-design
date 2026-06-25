import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# CVAE ARCHITECTURE
class CVAE(nn.Module):
    def __init__(self, recipe_dim=3, condition_dim=3, latent_dim=2):
        super(CVAE, self).__init__()
        
        # The Encoder
        self.encoder = nn.Sequential(
            nn.Linear(recipe_dim + condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Latent space parameters (Mean and Log-Variance)
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # The Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, recipe_dim)
        )

    def encode(self, x, c):
        # Concatenate inputs and conditions
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate latent vector and conditions
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


# LOSS FUNCTION

def loss_function(recon_x, x, mu, logvar):
    # Mean Squared Error
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# TRAINING LOOP
def train_model():
    print("Loading dataset...")
    df = pd.read_csv('data/recipes_and_properties.csv')

    recipe_data = df[['Temperature_C', 'Pulse_Time_ms', 'Plasma_Power_W']].values
    condition_data = df[['Target_GPC', 'Target_Refractive_Index', 'Target_Film_Stress_MPa']].values

    # Scale the data
    scaler_x = StandardScaler()
    scaler_c = StandardScaler()
    
    x_scaled = scaler_x.fit_transform(recipe_data)
    c_scaled = scaler_c.fit_transform(condition_data)

    x_tensor = torch.FloatTensor(x_scaled)
    c_tensor = torch.FloatTensor(c_scaled)

    # Initialize model and optimizer
    model = CVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Conditional VAE...")
    epochs = 300
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(x_tensor, c_tensor)
        
        # Calculate loss and Backpropagate
        loss = loss_function(recon_batch, x_tensor, mu, logvar)
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item() / len(df):.4f}")


    #  SAVE MODEL AND SCALERS
    os.makedirs('models', exist_ok=True)
    
    # Save network weights
    torch.save(model.state_dict(), 'models/cvae_weights.pth')
    
    # Save the scalers 
    with open('models/scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_x': scaler_x, 'scaler_c': scaler_c}, f)
        
    print("Training complete! Model and scalers securely saved to the models/ folder.")

if __name__ == "__main__":
    train_model()