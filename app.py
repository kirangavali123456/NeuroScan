import streamlit as st
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import new models
from model import MedicalVAE, MedicalGANGenerator, MedicalGANDiscriminator, MedicalTransformer, vae_loss_function
from utils import get_anomaly_heatmap, plot_comparison, calculate_metrics

# --- Config ---
IMAGE_SIZE = 128
BATCH_SIZE = 16 # Lowered batch size for Transformer/GAN memory safety
MODEL_DIR = "saved_models"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

st.set_page_config(page_title="NeuroScan Pro", layout="wide", page_icon="🩻")
st.title("🩻 NeuroScan Pro: Multi-Model Anomaly Detection")

# --- Session State for PERSISTENT GRAPHS ---
if 'history' not in st.session_state:
    st.session_state['history'] = {'VAE': [], 'GAN': [], 'ViT': []}

if 'dataset_path' not in st.session_state:
    st.session_state['dataset_path'] = os.path.join(os.getcwd(), "chest_xray")

# --- Helpers ---
def pick_folder():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

def get_data_loaders(root_dir, mode='train'):
    """Loads Data. mode='train' gets only NORMAL. mode='test' gets EVERYTHING."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(root_dir): return None, None
    
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    if mode == 'train':
        # Filter for NORMAL only
        idx = [i for i, label in enumerate(dataset.targets) if dataset.classes[label] == 'NORMAL']
        subset = torch.utils.data.Subset(dataset, idx)
        return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True), dataset.classes
    else:
        # Return full dataset for Testing/Metrics
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False), dataset.classes

def save_model(model, name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(MODEL_DIR, f"{name}_{timestamp}.pt")
    torch.save(model.state_dict(), path)
    return path

# --- Sidebar ---
st.sidebar.header("📂 Data & Config")
c1, c2 = st.sidebar.columns([3, 1])
with c2:
    if st.button("📂"):
        fp = pick_folder()
        if fp: st.session_state['dataset_path'] = fp
with c1:
    dataset_path = st.text_input("Dataset Path", value=st.session_state['dataset_path'])

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test") # Assumes Kaggle structure

# --- TAB LOGIC ---
tab1, tab2 = st.tabs(["🚀 Model Training", "🔎 Diagnostics & Benchmark"])

# ==========================================
# TAB 1: TRAINING
# ==========================================
with tab1:
    st.header("Train Models")
    model_type = st.selectbox("Select Model Architecture", ["VAE", "GAN", "Transformer (ViT)", "Train ALL Sequentially"])
    epochs = st.slider("Epochs", 1, 50, 10)
    lr = st.selectbox("Learning Rate", [1e-3, 1e-4, 2e-4])
    
    if st.button("Start Training"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader, _ = get_data_loaders(train_path, mode='train')
        
        if not loader:
            st.error("Could not load training data. Check path.")
            st.stop()

        models_to_train = []
        if model_type == "Train ALL Sequentially":
            models_to_train = ["VAE", "GAN", "ViT"]
        else:
            models_to_train = [model_type.split()[0]] # Extract "Transformer" from string
        
        progress_bar = st.progress(0)
        
        for m_name in models_to_train:
            st.subheader(f"Training {m_name}...")
            
            # Init Model
            if m_name == "VAE":
                model = MedicalVAE().to(device)
                opt = optim.Adam(model.parameters(), lr=lr)
            elif m_name == "GAN":
                gen = MedicalGANGenerator().to(device)
                disc = MedicalGANDiscriminator().to(device)
                opt_g = optim.Adam(gen.parameters(), lr=lr)
                opt_d = optim.Adam(disc.parameters(), lr=lr)
                criterion = nn.BCELoss()
            elif m_name in ["Transformer", "ViT"]:
                model = MedicalTransformer().to(device)
                opt = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

            # Train Loop
            history = []
            for epoch in range(epochs):
                epoch_loss = 0
                for x, _ in loader:
                    x = x.to(device)
                    
                    if m_name == "VAE":
                        opt.zero_grad()
                        recon, mu, logvar = model(x)
                        loss = vae_loss_function(recon, x, mu, logvar)
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item()
                        
                    elif m_name == "GAN":
                        # Train Disc
                        opt_d.zero_grad()
                        real_labels = torch.ones(x.size(0), 1).to(device)
                        fake_labels = torch.zeros(x.size(0), 1).to(device)
                        
                        d_real = disc(x)
                        d_loss_real = criterion(d_real, real_labels)
                        
                        fake_img = gen(x) # Autoencoder style GAN
                        d_fake = disc(fake_img.detach())
                        d_loss_fake = criterion(d_fake, fake_labels)
                        
                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        opt_d.step()
                        
                        # Train Gen
                        opt_g.zero_grad()
                        d_fake_preds = disc(fake_img)
                        # Gen wants Disc to think images are real (1) + Pixel Reconstruction
                        g_adv_loss = criterion(d_fake_preds, real_labels)
                        g_pixel_loss = F.mse_loss(fake_img, x)
                        g_loss = g_adv_loss + (100 * g_pixel_loss) # Weighted sum
                        g_loss.backward()
                        opt_g.step()
                        epoch_loss += g_loss.item()
                        
                    elif m_name in ["Transformer", "ViT"]:
                        opt.zero_grad()
                        recon = model(x)
                        loss = criterion(recon, x)
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)
                history.append(avg_loss)
                progress_bar.progress((epoch + 1) / epochs)
            
            # Save & Store History
            st.session_state['history'][m_name] = history # Store persistent history
            
            if m_name == "GAN": save_model(gen.cpu(), "GAN")
            else: save_model(model.cpu(), m_name)
            
            st.success(f"{m_name} Trained!")

    # --- PERSISTENT GRAPHS ---
    st.divider()
    st.subheader("Training Performance (Persistent)")
    if any(st.session_state['history'].values()):
        fig, ax = plt.subplots()
        for name, hist in st.session_state['history'].items():
            if hist: ax.plot(hist, label=f"{name} Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

# ==========================================
# TAB 2: DIAGNOSTICS
# ==========================================
with tab2:
    st.header("Comparative Diagnostics")
    
    # Loaders for 3 Models
    saved_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")), reverse=True)
    c1, c2, c3 = st.columns(3)
    
    vae_file = c1.selectbox("Load VAE", [f for f in saved_files if "VAE" in f] + ["None"])
    gan_file = c2.selectbox("Load GAN", [f for f in saved_files if "GAN" in f] + ["None"])
    vit_file = c3.selectbox("Load ViT", [f for f in saved_files if "Transformer" in f or "ViT" in f] + ["None"])
    
    if st.button("RUN BENCHMARK"):
        # Load Models
        models = {}
        if vae_file != "None":
            m = MedicalVAE()
            m.load_state_dict(torch.load(vae_file, map_location='cpu'))
            models['VAE'] = m
        if gan_file != "None":
            m = MedicalGANGenerator()
            m.load_state_dict(torch.load(gan_file, map_location='cpu'))
            models['GAN'] = m
        if vit_file != "None":
            m = MedicalTransformer()
            m.load_state_dict(torch.load(vit_file, map_location='cpu'))
            models['ViT'] = m
            
        if not models:
            st.error("Please load at least one model.")
        else:
            # 1. VISUAL COMPARISON
            st.subheader("Visual Reconstruction Comparison")
            test_loader, _ = get_data_loaders(test_path, mode='test') # Get mixed data
            
            # Get one random Pneumonia image
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            
            # Pick an image (Index 0 for simplicity)
            input_img = images[0].unsqueeze(0)
            
            recons = []
            titles = []
            
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    if name == "VAE": recon, _, _ = model(input_img)
                    else: recon = model(input_img)
                recons.append(recon)
                titles.append(f"{name} Recon")
            
            # Plot
            fig = plot_comparison(input_img, recons, titles)
            st.pyplot(fig)
            
            # 2. METRICS TABLE
            st.subheader("Model Accuracy & F1 Score (Test Set)")
            
            metrics_data = []
            progress_text = st.empty()
            
            for name, model in models.items():
                progress_text.text(f"Calculating metrics for {name}...")
                acc, f1 = calculate_metrics(model, test_loader)
                metrics_data.append({"Model": name, "Accuracy": f"{acc:.2%}", "F1 Score": f"{f1:.3f}"})
            
            progress_text.empty()
            df_metrics = pd.DataFrame(metrics_data)
            st.table(df_metrics)
