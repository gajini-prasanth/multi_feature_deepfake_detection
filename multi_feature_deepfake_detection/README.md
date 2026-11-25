# A Multi-Feature Deep Learning Framework for Detecting AI-Generated Images and Deepfake Media

## Overview
Three-stream forensic deep learning framework combining:
- **Stream A (Residual)**: 30 SRM high-pass kernels (90 residual channels) -> CNN -> 256-d.
- **Stream B (Frequency)**: 2D FFT (or block-wise 8×8 DCT) log-magnitude spectrum -> CNN -> 256-d.
- **Stream C (Content & Edge)**: RGB + constrained `BayarConv2d` + Sobel edge maps -> CNN -> 256-d.
- **Fusion**: 2-layer Cross Multi-Head Attention Fusion (compared against Concatenation and Dual Attention).

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Run Tests
```powershell
.\.venv\Scripts\pytest.exe tests -v
```

## Profile Ablation Models
```powershell
.\.venv\Scripts\python.exe main.py profile
```

## Launch Streamlit App
```powershell
.\.venv\Scripts\python.exe main.py app
```
