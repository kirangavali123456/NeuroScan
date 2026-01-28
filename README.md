# 🩻 NeuroScan: Generative Medical Anomaly Detection

> **Unsupervised Anomaly Detection for Chest X-Rays using Variational Autoencoders (VAE).**

NeuroScan is a deep learning application designed to assist in medical diagnostics. It uses a **Convolutional Variational Autoencoder (ConvVAE)** to learn the anatomical structure of *healthy* lungs. When presented with a pathological scan (e.g., Pneumonia), the model fails to reconstruct the anomaly, highlighting the diseased area in a heat map.

---

## 🌟 Key Features

* **Generative AI Core:** Custom PyTorch ConvVAE architecture trained on high-resolution medical images (128x128).
* **One-Class Classification:** Trains *only* on normal data, making it capable of detecting *any* anomaly (Pneumonia, Tuberculosis, etc.) without needing labeled diseased samples.
* **Visual Diagnostics:** Generates real-time "Anomaly Heatmaps" (Difference Maps) to pinpoint suspicious areas.
* **Model Management:**
    * **Auto-Save:** Automatically snapshots trained models with epoch/timestamp metadata.
    * **Model Loader:** Dropdown menu to load and reuse previous models instantly.
    * **Cross-Platform:** Models trained on GPU can be loaded on CPU for inference.
* **User-Friendly UI:** Built with **Streamlit** and includes a native **Folder Browser** to easily select datasets.

---

## 🛠️ Technical Stack

* **Deep Learning:** PyTorch, TorchVision
* **Interface:** Streamlit, Tkinter (for native file dialogs)
* **Data Processing:** NumPy, PIL
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/NeuroScan.git](https://github.com/yourusername/NeuroScan.git)
cd NeuroScan
```

### 2. Create Virtual Environment (Recommended)
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
*(Note: Ensure you have `tkinter` installed. It usually comes with Python, but on Linux you might need `sudo apt-get install python3-tk`)*

---

## 📂 Dataset Setup

This project is designed to work with the **Chest X-Ray Images (Pneumonia)** dataset.

1.  **Download Data:** [Kaggle Link](https://www.kaggle.com/datasets/paultimothythomas/chest-xray-pneumonia)
2.  **Extract:** Unzip the folder. You should have a structure like this:
    ```text
    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/ ...
    ```
3.  **Note Path:** Remember where you saved this folder. You will select it in the app.

---

## 🕹️ Usage Guide

### 1. Launch the App
```bash
streamlit run app.py
```

### 2. Configure Data
* On the sidebar, click the **📂 Button**.
* Select your extracted `chest_xray` folder.
* The app will automatically verify the path.

### 3. Train a Model (Tab 1)
* Go to the **"🚀 Train New Model"** tab.
* Set **Epochs** (Recommended: 20-50 for best results).
* Click **Start Training**.
* *The app will visualize the loss curve in real-time and auto-save the model upon completion.*

### 4. Run Diagnostics (Tab 2)
* Go to the **"🔎 Diagnostics & Loading"** tab.
* Select a saved model from the dropdown (or use the one you just trained).
* **Control Group Test:** Click "Scan Random Normal Lung". The heatmap should be mostly black (low error).
* **Test Group Test:** Click "Scan Random Pneumonia Lung". The heatmap will glow brightly over the infected areas, showing where the model detected anomalies.

---

## 🧠 Model Architecture

The `MedicalVAE` uses a deep convolutional network:

* **Input:** 128x128 Grayscale Image.
* **Encoder:** 4 Layers of `Conv2d` + `BatchNorm` + `LeakyReLU`. Compresses image to a dense latent vector.
* **Latent Space:** Uses the Reparameterization Trick ($z = \mu + \sigma \cdot \epsilon$) to enable generative sampling.
* **Decoder:** 4 Layers of `ConvTranspose2d` to upscale the latent vector back to 128x128.
* **Loss Function:** $MSE + \beta \cdot KLD$ (Reconstruction Loss + KL Divergence).

---

## 📂 Project Structure

```text
NeuroScan/
├── saved_models/       # Auto-created folder for .pt files
├── app.py              # Main Streamlit Application
├── model.py            # PyTorch Model Architecture
├── utils.py            # Visualization & Heatmap logic
├── requirements.txt    # Python dependencies
└── README.md           # Project Documentation
```

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Built for the purpose of demonstrating Generative AI in Healthcare.*
