import streamlit as st
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Import models & utils
from model import MedicalVAE, MedicalGANGenerator, MedicalGANDiscriminator, MedicalTransformer, vae_loss_function
from utils import plot_comparison, calculate_metrics

# --- Config ---
IMAGE_SIZE = 128
BATCH_SIZE = 16 
MODEL_DIR = "saved_models"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

st.set_page_config(page_title="NeuroScan Pro", layout="wide", page_icon="🩻")
st.title("🩻 NeuroScan Pro: Multi-Model Anomaly Detection")

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state['history'] = {'VAE': [], 'GAN': [], 'ViT': []}

if 'leaderboard' not in st.session_state:
    st.session_state['leaderboard'] = []

# Persistent state for Tab 2 Benchmark Table
if 'benchmark_data' not in st.session_state:
    st.session_state['benchmark_data'] = None

if 'dataset_path' not in st.session_state:
    st.session_state['dataset_path'] = os.path.join(os.getcwd(), "chest_xray")

# --- Helpers ---
def pick_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_path
    except:
        return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data_loaders(root_dir, mode='train'):
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
        subset = Subset(dataset, idx)
        return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True), dataset.classes
    else:
        # Return full dataset for Testing/Metrics
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False), dataset

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
test_path = os.path.join(dataset_path, "test")

# --- TAB LOGIC ---
tab1, tab2 = st.tabs(["🚀 Training & Metrics", "🔎 Comparative Diagnostics"])

# ==========================================
# TAB 1: TRAINING & METRICS
# ==========================================
with tab1:
    col_ctrl, col_metrics = st.columns([2, 1])

    # --- LEFT: CONTROLS ---
    with col_ctrl:
        st.header("1. Train New Model")
        model_type = st.selectbox("Select Model Architecture", ["VAE", "GAN", "Transformer (ViT)", "Train ALL Sequentially"])
        epochs = st.slider("Epochs", 1, 50, 10)
        lr = st.selectbox("Learning Rate", [1e-3, 1e-4, 2e-4])
        
        start_btn = st.button("Start Training")

    # --- RIGHT: LEADERBOARD ---
    with col_metrics:
        st.subheader("🏆 Live Leaderboard")
        if st.session_state['leaderboard']:
            st.dataframe(pd.DataFrame(st.session_state['leaderboard']))
            if st.button("Clear Leaderboard"):
                st.session_state['leaderboard'] = []
                st.rerun()
        else:
            st.info("Train a model to see metrics here.")

    # --- TRAINING LOOP ---
    if start_btn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader, _ = get_data_loaders(train_path, mode='train')
        test_loader, _ = get_data_loaders(test_path, mode='test') 
        
        if not loader:
            st.error("Could not load training data. Check path.")
            st.stop()

        models_to_train = []
        if model_type == "Train ALL Sequentially":
            models_to_train = ["VAE", "GAN", "ViT"]
        else:
            models_to_train = [model_type.split()[0]]
        
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
                model = gen 
            elif m_name in ["Transformer", "ViT"]:
                model = MedicalTransformer().to(device)
                opt = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

            # SHOW PARAMS
            params = count_parameters(model)
            st.info(f"🧠 **{m_name} Parameters:** {params:,}")

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
                        # Disc
                        opt_d.zero_grad()
                        real_labels = torch.ones(x.size(0), 1).to(device)
                        fake_labels = torch.zeros(x.size(0), 1).to(device)
                        d_real = disc(x)
                        d_loss_real = criterion(d_real, real_labels)
                        fake_img = gen(x)
                        d_fake = disc(fake_img.detach())
                        d_loss_fake = criterion(d_fake, fake_labels)
                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        opt_d.step()
                        # Gen
                        opt_g.zero_grad()
                        d_fake_preds = disc(fake_img)
                        g_adv_loss = criterion(d_fake_preds, real_labels)
                        g_pixel_loss = F.mse_loss(fake_img, x)
                        g_loss = g_adv_loss + (100 * g_pixel_loss)
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
            
            st.session_state['history'][m_name] = history
            
            if m_name == "GAN": save_model(gen.cpu(), "GAN")
            else: save_model(model.cpu(), m_name)

            # --- AUTO METRICS ---
            model.cpu()
            acc, f1 = calculate_metrics(model, test_loader)
            st.session_state['leaderboard'].append({
                "Model": m_name,
                "Accuracy": f"{acc:.2%}",
                "F1 Score": f"{f1:.3f}",
                "Params": f"{params:,}",
                "Timestamp": datetime.now().strftime("%H:%M")
            })
            
            st.success(f"✅ {m_name} Trained & Saved!")
            # --- PERSISTENT GRAPHS (LOG SCALE) ---
    st.divider()
    st.subheader("Training Performance (Logarithmic Scale)")
    if any(st.session_state['history'].values()):
        fig, ax = plt.subplots()
        for name, hist in st.session_state['history'].items():
            if hist: ax.plot(hist, label=f"{name} Loss")
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (Log Scale)")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        st.pyplot(fig)

# ==========================================
# TAB 2: DIAGNOSTICS & BENCHMARK
# ==========================================
with tab2:
    st.header("Comparative Diagnostics")
    
    saved_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")), reverse=True)
    c1, c2, c3 = st.columns(3)
    
    vae_file = c1.selectbox("Load VAE", [f for f in saved_files if "VAE" in f] + ["None"])
    gan_file = c2.selectbox("Load GAN", [f for f in saved_files if "GAN" in f] + ["None"])
    vit_file = c3.selectbox("Load ViT", [f for f in saved_files if "Transformer" in f or "ViT" in f] + ["None"])
    
    st.divider()

    def load_selected_models():
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
        return models

    # ==========================
    # SECTION 1: PERFORMANCE TABLE
    # ==========================
    st.subheader("1. Model Performance Benchmark")
    
    if st.button("📊 Run Performance Benchmark", use_container_width=True):
        models = load_selected_models()
        if not models:
            st.error("Please load at least one model.")
        else:
            test_loader, _ = get_data_loaders(test_path, mode='test')
            
            with st.spinner("Calculating Metrics..."):
                metrics_data = []
                for name, model in models.items():
                    acc, f1 = calculate_metrics(model, test_loader)
                    params = count_parameters(model)
                    metrics_data.append({
                        "Model": name, 
                        "Accuracy": f"{acc:.2%}", 
                        "F1 Score": f"{f1:.3f}",
                        "Parameters": f"{params:,}"
                    })
                
                # Save to session state so it doesn't disappear
                st.session_state['benchmark_data'] = pd.DataFrame(metrics_data)

    # DISPLAY TABLE (PERSISTENT)
    if st.session_state['benchmark_data'] is not None:
        st.table(st.session_state['benchmark_data'])
        if st.button("Clear Table"):
            st.session_state['benchmark_data'] = None
            st.rerun()

    st.divider()

    # ==========================
    # SECTION 2: VISUAL INSPECTION
    # ==========================
    st.subheader("2. Visual Reconstruction Inspection")
    
    if st.button("👁️ Run Visual Inspection (Random Batch)", use_container_width=True):
        models = load_selected_models()
        if not models:
            st.error("Please load at least one model.")
        else:
            _, full_test_dataset = get_data_loaders(test_path, mode='test')
            
            def get_random_image(target_label_name):
                target_idx = full_test_dataset.class_to_idx[target_label_name]
                indices = [i for i, label in enumerate(full_test_dataset.targets) if label == target_idx]
                rand_idx = np.random.choice(indices)
                img, _ = full_test_dataset[rand_idx]
                return img.unsqueeze(0)

            col_normal, col_pneumonia = st.columns(2)
            
            # A. NORMAL CASE
            with col_normal:
                st.info("🟢 **Control Case: NORMAL**")
                img_normal = get_random_image("NORMAL")
                
                recons_n = []
                titles_n = []
                for name, model in models.items():
                    model.eval()
                    with torch.no_grad():
                        recon = model(img_normal) if name != "VAE" else model(img_normal)[0]
                    recons_n.append(recon)
                    titles_n.append(name)
                
                fig_n = plot_comparison(img_normal, recons_n, titles_n)
                st.pyplot(fig_n)

            # B. PNEUMONIA CASE
            with col_pneumonia:
                st.error("🔴 **Test Case: PNEUMONIA**")
                img_pneu = get_random_image("PNEUMONIA")
                
                recons_p = []
                titles_p = []
                for name, model in models.items():
                    model.eval()
                    with torch.no_grad():
                        recon = model(img_pneu) if name != "VAE" else model(img_pneu)[0]
                    recons_p.append(recon)
                    titles_p.append(name)
                
                fig_p = plot_comparison(img_pneu, recons_p, titles_p)
                st.pyplot(fig_p)
            
            st.caption("Each click loads a new random set of images.")
