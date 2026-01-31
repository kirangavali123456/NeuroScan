import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. VAE (Existing)
# ==========================================
class MedicalVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(MedicalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid()
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

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# ==========================================
# 2. GAN (Adversarial Autoencoder)
# ==========================================
class MedicalGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super(MedicalGANGenerator, self).__init__()
        # Using a U-Net style generator or simple Autoencoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32768, latent_dim) # 128*16*16 flattened
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32768),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class MedicalGANDiscriminator(nn.Module):
    def __init__(self):
        super(MedicalGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16384, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. Vision Transformer (ViT-AE)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, Embed, H/P, W/P)
        x = x.flatten(2)  # (B, Embed, N_Patches)
        x = x.transpose(1, 2)  # (B, N_Patches, Embed)
        return x

class MedicalTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Simple Decoder (Project back to pixels)
        self.decoder_proj = nn.Linear(embed_dim, patch_size*patch_size)
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. Patchify & Embed
        patches = self.patch_embed(x) 
        
        # 2. Transformer
        encoded = self.transformer_encoder(patches)
        
        # 3. Reconstruct Patches
        rec_patches = self.decoder_proj(encoded) # (B, N_Patches, P*P)
        
        # 4. Reshape back to image (simplified unpatchify)
        # This part requires careful reshaping
        rec_patches = rec_patches.transpose(1, 2) # (B, P*P, N)
        rec_img = F.fold(rec_patches, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        
        return torch.sigmoid(rec_img)
