import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(4, latent_dim)
        self.fc_logvar = nn.Linear(4, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def plane():
    # Parameters
    input_dim = 4
    hidden_dim = 4
    latent_dim = 4
    epochs = 20
    batch_size = 128
    lr = 1e-3

    train_loader = DataLoader(torch.load('data/offline-200.pth')[:-48], batch_size=batch_size, shuffle=True)

    # Initialize VAE model and optimizer
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Loss function
    def vae_loss(x, x_recon, mu, logvar):
        recon_loss = nn.BCELoss(reduction='sum')(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    # Training loop
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, actions, old_log_probs, returns, advantages in train_loader:
            x = x.to(device)
            # Forward pass
            x_recon, mu, logvar = vae(x)
            loss = vae_loss(x, x_recon, mu, logvar)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
