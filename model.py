import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(MedicalVAE, self).__init__()
        
        # ===========================
        # Encoder (128x128 Input)
        # ===========================
        self.encoder = nn.Sequential(
            # Input: 1 x 128 x 128
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Flatten() # 256 * 8 * 8 = 16384
        )
        
        # Latent Space
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 16384)
        
        # ===========================
        # Decoder
        # ===========================
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> 128x128
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_proj = self.decoder_input(z)
        return self.decoder(z_proj), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss
