This is a Setup Guide for Billboard Ad Allocation Environment


# Installation Instructions

**Step 1: Create Virtual Environment*

```bash
# Create virtual environment
python3 -m venv venv       # Recommended Python 3.12

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

---

**Step 2: Upgrade pip*

```bash
pip install --upgrade pip
```

---

**Step 3: Install PyTorch (Installing first due to combat mismatch issues that may occur)*

Option A: With CUDA 12.1 Support (GPU)
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Option C: Different CUDA Version
Check your CUDA version using:
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```
Then install matching PyTorch from: https://pytorch.org/get-started/locally/

---

**Step 4: Install Core Dependencies*

```bash
pip install torch-geometric==2.7.0
pip install tianshou==1.2.0
pip install gymnasium==0.28.1
pip install pettingzoo==1.24.3
```

---

**Step 5: Install Scientific Computing Packages*

```bash
pip install numpy==1.26.4
pip install scipy==1.16.3
pip install pandas==2.3.3
pip install networkx==3.6.1
```

---

**Step 6: Install Visualization & Logging*

```bash
pip install tensorboard==2.20.0
pip install matplotlib==3.10.8
pip install seaborn==0.13.2
```

---

