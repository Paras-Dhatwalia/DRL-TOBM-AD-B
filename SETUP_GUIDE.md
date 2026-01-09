# 🚀 Complete Setup Guide for Billboard Allocation Environment

## 📦 **Required Packages**

### **Minimum Required Files for Your New Folder**

Copy these files to your new project folder:

**1. Environment Files:**
- `optimized_env.py` (or `optimized_env_clean_logs.py` for clean logging)
- `models.py`
- `wrappers.py` (if using graph wrappers)

**2. Training Scripts:**
- `training_na_2.py` (or `training_na_clean_logs.py` for clean logs)
- `training_ea_2.py` (if using EA mode)

**3. Data Files:**
- `BB_NYC.csv` or `Billboard_NYC.csv` (billboard data)
- `Advertiser_5.csv` or `Advertiser_NYC2.csv` (advertiser data)
- `TJ_NYC.csv` (trajectory/user location data)

**4. Configuration:**
- `requirements.txt` or `EXACT_REQUIREMENTS.txt` (this guide)

---

## 🔧 **Installation Instructions**

### **Step 1: Create Virtual Environment**

```bash
# Navigate to your new project folder
cd /path/to/your/new/folder

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

---

### **Step 2: Upgrade pip**

```bash
pip install --upgrade pip
```

---

### **Step 3: Install PyTorch (CRITICAL - Do This First!)**

#### **Option A: With CUDA 12.1 Support (GPU)**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### **Option B: CPU Only (No GPU)**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

#### **Option C: Different CUDA Version**
Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```

Then install matching PyTorch from: https://pytorch.org/get-started/locally/

---

### **Step 4: Install Core Dependencies**

```bash
pip install torch-geometric==2.7.0
pip install tianshou==1.2.0
pip install gymnasium==0.28.1
pip install pettingzoo==1.24.3
```

---

### **Step 5: Install Scientific Computing Packages**

```bash
pip install numpy==1.26.4
pip install scipy==1.16.3
pip install pandas==2.3.3
pip install networkx==3.6.1
```

---

### **Step 6: Install Visualization & Logging**

```bash
pip install tensorboard==2.20.0
pip install matplotlib==3.10.8
pip install seaborn==0.13.2
```

---

### **Step 7: Install Utilities**

```bash
pip install psutil==7.2.1
```

---

## ✅ **Verification**

### **Check Installation**

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python3 -c "import tianshou; print(f'Tianshou: {tianshou.__version__}')"
python3 -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.5.1+cu121
CUDA Available: True
PyG: 2.7.0
Tianshou: 1.2.0
Gymnasium: 0.28.1
```

---

### **Test Environment**

```bash
# Quick test to make sure environment loads
python3 -c "
from optimized_env import OptimizedBillboardEnv, EnvConfig

env = OptimizedBillboardEnv(
    billboard_csv='BB_NYC.csv',
    advertiser_csv='Advertiser_5.csv',
    trajectory_csv='TJ_NYC.csv',
    action_mode='na',
    config=EnvConfig(max_events=10)
)

obs, info = env.reset()
print('✅ Environment loaded successfully!')
print(f'Billboards: {env.n_nodes}')
print(f'Observation keys: {list(obs.keys())}')
"
```

---

## 📋 **Complete Package List (For Reference)**

### **CRITICAL PACKAGES (Must match versions)**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.5.1+cu121 | Deep learning framework |
| `torch-geometric` | 2.7.0 | Graph neural networks |
| `tianshou` | 1.2.0 | RL framework (PPO) |
| `gymnasium` | 0.28.1 | RL environment API |
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.3.3 | Data manipulation |

### **IMPORTANT PACKAGES (Should match versions)**

| Package | Version | Purpose |
|---------|---------|---------|
| `scipy` | 1.16.3 | Scientific computing |
| `networkx` | 3.6.1 | Graph utilities |
| `tensorboard` | 2.20.0 | Training visualization |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical plots |
| `pettingzoo` | 1.24.3 | Multi-agent support |

### **UTILITY PACKAGES (Version flexible)**

| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | 7.2.1 | System monitoring |
| `tqdm` | 4.67.1 | Progress bars |
| `pillow` | 12.0.0 | Image processing |

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: CUDA Version Mismatch**

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
Check CUDA version and install matching PyTorch:
```bash
nvidia-smi  # Check your CUDA version
# Then install matching PyTorch from pytorch.org
```

---

### **Issue 2: torch-geometric Installation Fails**

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Solution:**
Install PyTorch FIRST, then torch-geometric:
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0
```

---

### **Issue 3: ImportError for tianshou**

**Error:**
```
ModuleNotFoundError: No module named 'tianshou'
```

**Solution:**
Ensure gymnasium is installed first:
```bash
pip install gymnasium==0.28.1
pip install tianshou==1.2.0
```

---

### **Issue 4: NumPy Version Conflict**

**Error:**
```
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
```

**Solution:**
Reinstall numpy:
```bash
pip uninstall numpy
pip install numpy==1.26.4
```

---

## 🎯 **One-Command Installation (After PyTorch)**

After installing PyTorch with CUDA support, run this single command:

```bash
pip install torch-geometric==2.7.0 tianshou==1.2.0 gymnasium==0.28.1 pettingzoo==1.24.3 numpy==1.26.4 scipy==1.16.3 pandas==2.3.3 networkx==3.6.1 tensorboard==2.20.0 matplotlib==3.10.8 seaborn==0.13.2 psutil==7.2.1
```

---

## 📁 **Minimal File Structure**

Your new folder should look like this:

```
your-new-folder/
├── venv/                          # Virtual environment
├── optimized_env.py               # Environment (with budget tracking)
├── models.py                      # GNN model architecture
├── training_na_clean_logs.py      # Training script (clean logs)
├── BB_NYC.csv                     # Billboard data
├── Advertiser_5.csv               # Advertiser data
├── TJ_NYC.csv                     # Trajectory data
├── EXACT_REQUIREMENTS.txt         # This file (for reference)
└── models/                        # Will be created by training
    └── ppo_billboard_na_clean.pt  # Saved model
```

**Optional files:**
- `wrappers.py` - If using custom observation wrappers
- `training_ea_2.py` - If using EA mode
- `test_*.py` - Testing scripts
- `*.md` - Documentation

---

## 🏃 **Quick Start**

```bash
# 1. Setup
cd your-new-folder
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else
pip install torch-geometric==2.7.0 tianshou==1.2.0 gymnasium==0.28.1 pettingzoo==1.24.3 numpy==1.26.4 scipy==1.16.3 pandas==2.3.3 networkx==3.6.1 tensorboard==2.20.0 matplotlib==3.10.8 seaborn==0.13.2 psutil==7.2.1

# 4. Verify
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 5. Train!
python training_na_clean_logs.py --epochs 50 --log-level INFO
```

---

## 🔍 **Package Dependency Tree**

```
billboard-allocation-env
│
├── torch (2.5.1+cu121)
│   ├── numpy (1.26.4)
│   ├── sympy
│   ├── jinja2
│   └── CUDA libraries (nvidia-*)
│
├── torch-geometric (2.7.0)
│   ├── torch (must be installed first!)
│   ├── torch-scatter (auto-installed)
│   ├── torch-sparse (auto-installed)
│   └── scipy (1.16.3)
│
├── tianshou (1.2.0)
│   ├── gymnasium (0.28.1)
│   ├── numpy (1.26.4)
│   ├── tqdm
│   └── tensorboard (2.20.0)
│
├── pandas (2.3.3)
│   ├── numpy (1.26.4)
│   └── python-dateutil
│
└── visualization
    ├── matplotlib (3.10.8)
    ├── seaborn (0.13.2)
    └── tensorboard (2.20.0)
```

---

## ✅ **Final Checklist**

Before starting training:

- [ ] Virtual environment created and activated
- [ ] PyTorch installed with CUDA support (check with `nvidia-smi`)
- [ ] All packages installed (verify with pip list)
- [ ] Data files present (BB_NYC.csv, Advertiser_5.csv, TJ_NYC.csv)
- [ ] Environment file present (optimized_env.py with budget tracking)
- [ ] Training script present (training_na_clean_logs.py)
- [ ] Environment loads successfully (test with quick test above)
- [ ] GPU detected (python -c "import torch; print(torch.cuda.is_available())")

---

**You're ready to train!** 🚀

Run:
```bash
python training_na_clean_logs.py --epochs 50 --log-level INFO
```

Expected first epoch reward: **-15 to -10** (much better than old -44!)
