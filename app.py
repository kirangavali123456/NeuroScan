import streamlit as st
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

from model import MedicalVAE, loss_function
from utils import get_anomaly_heatmap, plot_anomaly_detection

# --- Config ---
IMAGE_SIZE = 128
BATCH_SIZE = 32
MODEL_DIR = "saved_models"

# Ensure directories exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

st.set_page_config(page_title="NeuroScan: Medical Edition", layout="wide", page_icon="🩻")

st.title("🩻 NeuroScan: X-Ray Anomaly Detection")
st.markdown("Train or Load a VAE model to detect **Pneumonia** anomalies in Chest X-Rays.")

# --- Helper: Folder Picker ---
def pick_folder():
    """Opens a native OS dialog to select a folder."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.wm_attributes('-topmost', 1) # Ensure popup is on top
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

# --- Session State Init ---
if 'dataset_path' not in st.session_state:
    st.session_state['dataset_path'] = os.path.join(os.getcwd(), "chest_xray")

# --- Sidebar ---
st.sidebar.header("Data Configuration")

# Layout: Text Input + Browse Button
col1, col2 = st.sidebar.columns([3, 1])

with col2:
    st.write("") # Spacer to align button
    st.write("") 
    if st.button("📂"):
        selected_folder = pick_folder()
        if selected_folder:
            st.session_state['dataset_path'] = selected_folder

with col1:
    dataset_path = st.text_input("Path to 'chest_xray' folder", value=st.session_state['dataset_path'])

# Update train path based on selection
train_path = os.path.join(dataset_path, "train")

# --- Helper Functions ---
def get_data_loaders(root_dir):
    """Loads Normal images for training/testing."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(root_dir):
        return None, None, None

    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    class_to_idx = full_dataset.class_to_idx
    
    if 'NORMAL' not in class_to_idx:
        return None, None, None
        
    normal_idx = class_to_idx['NORMAL']
    normal_indices = [i for i, label in enumerate(full_dataset.targets) if label == normal_idx]
    normal_subset = torch.utils.data.Subset(full_dataset, normal_indices)
    loader = DataLoader(normal_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    return loader, full_dataset, class_to_idx

def save_model(model, epochs):
    """Saves model state_dict with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"medical_vae_e{epochs}_{timestamp}.pt"
    filepath = os.path.join(MODEL_DIR, filename)
    torch.save(model.state_dict(), filepath)
    return filename

def load_model_from_file(filepath):
    """Loads state_dict into a fresh model instance."""
    model = MedicalVAE(latent_dim=128)
    state_dict = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Tabs for Workflow ---
tab1, tab2 = st.tabs(["🚀 Train New Model", "🔎 Diagnostics & Loading"])

# ==========================================
# TAB 1: TRAINING
# ==========================================
with tab1:
    st.header("Train a New Anomaly Detector")
    st.info(f"Training Data Location: `{train_path}`")
    
    epochs = st.slider("Training Epochs", 5, 100, 20)
    lr = st.selectbox("Learning Rate", [1e-3, 1e-4], index=0)
    
    if st.button("Start Training"):
        if not os.path.exists(train_path):
            st.error(f"❌ Path not found: {train_path}\nPlease select the correct 'chest_xray' folder using the Browse button.")
        else:
            train_loader, _, _ = get_data_loaders(train_path)
            
            if train_loader is None:
                st.error("❌ 'NORMAL' folder not found inside train directory. Check dataset structure.")
            else:
                model = MedicalVAE(latent_dim=128)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
                loss_history = []
                
                model.train()
                for epoch in range(epochs):
                    epoch_loss = 0
                    for batch_idx, (data, _) in enumerate(train_loader):
                        data = data.to(device)
                        optimizer.zero_grad()
                        recon, mu, logvar = model(data)
                        loss = loss_function(recon, data, mu, logvar)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    avg_loss = epoch_loss / len(train_loader.dataset)
                    loss_history.append(avg_loss)
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.2f}")
                    loss_chart.line_chart(loss_history)
                
                saved_name = save_model(model.cpu(), epochs)
                st.success(f"✅ Training Complete! Model saved as: `{saved_name}`")
                
                st.session_state['active_model'] = model
                st.session_state['train_path'] = train_path

# ==========================================
# TAB 2: DIAGNOSTICS & LOADING
# ==========================================
with tab2:
    st.header("Select Model for Diagnosis")
    
    saved_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")), key=os.path.getmtime, reverse=True)
    
    use_saved = st.checkbox("Load a Pre-Trained Model", value=True)
    
    if use_saved:
        if not saved_files:
            st.warning("No saved models found in `saved_models/`.")
        else:
            selected_file = st.selectbox("Select Model File", saved_files)
            if st.button("📂 Load Selected Model"):
                try:
                    loaded_model = load_model_from_file(selected_file)
                    st.session_state['active_model'] = loaded_model
                    st.session_state['train_path'] = train_path
                    st.success(f"Loaded: {os.path.basename(selected_file)}")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
    
    st.divider()
    
    if 'active_model' in st.session_state:
        st.subheader("Run Diagnostics")
        model = st.session_state['active_model']
        
        _, full_dataset, class_map = get_data_loaders(train_path)
        
        if full_dataset:
            c1, c2 = st.columns(2)
            
            with c1:
                st.info("**Control Group: NORMAL**\n\nExpected Result: The Difference Map should be dark. This confirms the model recognizes healthy tissue.")
                if st.button("Scan Random Normal Lung"):
                    normal_idx = class_map['NORMAL']
                    indices = [i for i, label in enumerate(full_dataset.targets) if label == normal_idx]
                    rand_idx = np.random.choice(indices)
                    img, label = full_dataset[rand_idx]
                    
                    with torch.no_grad():
                        recon, _, _ = model(img.unsqueeze(0))
                    
                    orig, rec, diff = get_anomaly_heatmap(img, recon)
                    fig = plot_anomaly_detection(orig, rec, diff)
                    st.pyplot(fig)

            with c2:
                st.warning("**Test Group: PNEUMONIA**\n\nExpected Result: The Difference Map should 'glow' (bright spots). These are anomalies the model failed to reconstruct.")
                if st.button("Scan Random Pneumonia Lung"):
                    pneu_idx = class_map.get('PNEUMONIA')
                    if pneu_idx is None:
                        st.warning("PNEUMONIA folder not found.")
                    else:
                        indices = [i for i, label in enumerate(full_dataset.targets) if label == pneu_idx]
                        if not indices:
                            st.warning("No Pneumonia images found.")
                        else:
                            rand_idx = np.random.choice(indices)
                            img, label = full_dataset[rand_idx]
                            
                            with torch.no_grad():
                                recon, _, _ = model(img.unsqueeze(0))
                            
                            orig, rec, diff = get_anomaly_heatmap(img, recon)
                            fig = plot_anomaly_detection(orig, rec, diff)
                            st.pyplot(fig)
    else:
        st.info("👈 Please Train a model or Load one to begin diagnostics.")
