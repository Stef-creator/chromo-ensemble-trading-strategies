# unconditional_flow.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal

# === Step 1. Load data and prepare returns ===

df = pd.read_csv('data/AAPL_technical_indicators.csv')
df['Return'] = df['Close'].pct_change()
returns = df['Return'].dropna().values
returns = returns.reshape(-1, 1).astype('float32')

# Convert to tensor
returns_tensor = torch.tensor(returns)

# === Step 2. Define RealNVP coupling layer ===

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, dim):
        super(RealNVPCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, dim),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, dim)
        )

    def forward(self, x, reverse=False):
        s = self.scale_net(x)
        t = self.translate_net(x)
        if not reverse:
            y = x * torch.exp(s) + t
            log_det_jacobian = s.sum(-1)
            return y, log_det_jacobian
        else:
            y = (x - t) * torch.exp(-s)
            log_det_jacobian = -s.sum(-1)
            return y, log_det_jacobian

# === Step 3. Define simple RealNVP model ===

class RealNVP(nn.Module):
    def __init__(self, dim, num_coupling_layers=4):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList([RealNVPCouplingLayer(dim) for _ in range(num_coupling_layers)])
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def inverse(self, z):
        log_det_jacobian = 0
        for layer in reversed(self.layers):
            z, ldj = layer(z, reverse=True)
            log_det_jacobian += ldj
        return z, log_det_jacobian

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        log_prob_z = self.base_dist.log_prob(z)
        return log_prob_z + log_det_jacobian

# === Step 4. Train flow model ===

model = RealNVP(dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = -model.log_prob(returns_tensor).mean()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# === Step 5. Sample from flow and plot ===

with torch.no_grad():
    z = model.base_dist.sample((1000,))
    samples, _ = model.inverse(z)

# Convert to numpy
samples = samples.numpy()

# Plot empirical vs flow samples
plt.figure(figsize=(10,5))
plt.hist(returns, bins=50, density=True, alpha=0.6, label='Empirical')
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Flow Samples')
plt.legend()
plt.title("Empirical vs Flow Sampled Return Distributions")
plt.savefig("plots/unconditional_flow_returns.png")
plt.show()
