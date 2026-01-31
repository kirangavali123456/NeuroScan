# 🩻 NeuroScan Pro: Multi-Model Anomaly Detection

> **Advanced Unsupervised Anomaly Detection for Medical Imaging using VAEs, GANs, and Vision Transformers.**

NeuroScan Pro is a state-of-the-art benchmarking workbench for medical AI. It uses **One-Class Classification** to detect pathologies (like Pneumonia) by training strictly on healthy tissue.

Unlike standard classifiers, NeuroScan Pro learns the "manifold of health." When presented with disease, the models fail to reconstruct the anatomy correctly, creating a "Difference Map" that highlights the tumor or infection.

---

## 🌟 New Pro Features

* **Three Architectures:**
    * **ConvVAE (Variational Autoencoder):** The classic probabilistic baseline.
    * **GAN (Generative Adversarial Network):** Uses adversarial training to generate sharper, more realistic healthy tissue reconstructions.
    * **ViT (Vision Transformer):** Uses Self-Attention mechanisms to capture long-range dependencies in anatomical structures.
* **⚔️ Benchmark Suite:**
    * **Visual Comparison:** See how VAE, GAN, and ViT reconstruct the same X-Ray side-by-side.
    * **Anomaly Localization:** Compare "Difference Maps" to see which model best highlights the disease.
    * **Metric Table:** Auto-calculates **Accuracy** and **F1-Score** for all loaded models on the Test set.
* **Persistent History:** Training loss graphs are saved in memory, allowing you to train Model A, then Model B, and compare their learning curves on the same plot.
* **Auto-Save & Load:** Models are automatically timestamped and saved. The "Diagnostics" tab lets you mix and match versions (e.g., "Load VAE from yesterday vs. GAN from today").

---

## 🛠️ Technical Stack

* **Core:** PyTorch, TorchVision
* **UI:** Streamlit
* **Algorithms:**
    * **VAE:** Standard Encoder-Decoder with KL Divergence loss.
    * **GAN:** Autoencoder-style Generator with a PatchGAN Discriminator.
    * **Transformer:** Patch-based ViT Encoder with a linear projection Decoder.
* **Metrics:** Scikit-Learn (Accuracy, F1).

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/NeuroScanPro.git](https://github.com/yourusername/NeuroScanPro.git)
cd NeuroScanPro
```

### 2. Set up Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

**Required:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothythomas/chest-xray-pneumonia) from Kaggle.

1.  Download and Unzip.
2.  Ensure folder structure:
    ```text
    chest_xray/
    ├── train/
    │   ├── NORMAL/     <-- Training Data (Healthy)
    │   └── PNEUMONIA/  <-- Ignored during training
    └── test/           <-- Used for Benchmarking (Contains Both)
    ```
3.  **Note Path:** You will select this folder using the "📂" button in the app.

---

## 🕹️ Usage Guide

### 1. Launch the Workbench
```bash
streamlit run app.py
```

### 2. Train Models (Tab 1)
* **Select Architecture:** Choose VAE, GAN, Transformer, or "Train ALL Sequentially".
* **Configure:** Set Epochs (Rec: 20+) and Learning Rate.
* **Train:** Click Start.
    * *The Loss Graph will update in real-time and persists across different training runs.*
    * *Models are auto-saved to `saved_models/`.*

### 3. Run Diagnostics & Benchmark (Tab 2)
* **Load Models:** Use the dropdowns to select specific `.pt` files for VAE, GAN, and ViT. (Select "None" if you only want to test one).
* **Click "RUN BENCHMARK":**
    * **Visuals:** Shows the Input X-Ray, Reconstructions, and **Difference Maps** (Heatmaps) for all loaded models.
    * **Metrics:** Displays a table with **Accuracy** and **F1 Score** to numerically prove which architecture performs best.

---

## 🧠 Model Architectures

| Model | Mechanism | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **VAE** | Compresses input to Gaussian latent space. | Smooth, stable training. | Blurry reconstructions. |
| **GAN** | Generator fights Discriminator. | Sharp, realistic details. | Unstable training (Mode Collapse). |
| **ViT** | Splits image into 16x16 patches + Self-Attention. | Understands global structure. | Data hungry; heavy compute. |

---

## 📂 Project Structure

```text
NeuroScanPro/
├── saved_models/       # Auto-created folder for trained weights
├── app.py              # Main Benchmark Dashboard
├── model.py            # PyTorch Architectures (VAE, GAN, ViT)
├── utils.py            # Metrics, Heatmaps & Plotting Logic
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---
## Images
<img width="1286" height="614" alt="neuroscan train models" src="https://github.com/user-attachments/assets/802c6338-368c-45ca-963a-3e766a14dbe8" />
<img width="1288" height="356" alt="diagnostic and benchmark" src="https://github.com/user-attachments/assets/971569a5-e788-4be1-8a50-a2b4de93e60a" />
<img width="1001" height="545" alt="visual reconstruction" src="https://github.com/user-attachments/assets/939512f6-5e38-4492-a668-fc2d586738a1" />
<img width="1002" height="193" alt="accuracy and f1 score" src="https://github.com/user-attachments/assets/8c59f6a8-86bf-4eb2-beb8-d5880c5d114c" />


