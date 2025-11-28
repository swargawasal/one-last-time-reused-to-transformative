# 🚀 Google Colab Setup Guide (T4 GPU)

## ⚡ Quick Start (3 Steps)

### Step 1: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

### Step 2: Create `.env` File

```python
%%writefile .env
# ==================== CORE SETTINGS ====================
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# ==================== PERFORMANCE ====================
# FAST_MODE: yes = FFmpeg only (10x faster), no = AI enhancement
FAST_MODE=no
AI_FAST_MODE=yes
COMPUTE_MODE=auto
ENHANCEMENT_LEVEL=high

# ==================== TRANSFORMATIVE FEATURES ====================
ADD_TEXT_OVERLAY=yes
TEXT_OVERLAY_TEXT=🔥 VIRAL
TEXT_OVERLAY_POSITION=bottom
TEXT_OVERLAY_STYLE=modern

ADD_COLOR_GRADING=yes
COLOR_FILTER=cinematic
COLOR_INTENSITY=0.5

ADD_SPEED_RAMPING=yes
SPEED_VARIATION=0.15

FORCE_AUDIO_REMIX=yes

# ==================== COMPILATION ====================
COMPILATION_BATCH_SIZE=6
SEND_TO_YOUTUBE=off
TARGET_RESOLUTION=1080:1920
TRANSITION_DURATION=1.0
TRANSITION_INTERVAL=10
```

> **⚠️ IMPORTANT:** Replace `YOUR_BOT_TOKEN_HERE` and `YOUR_GEMINI_API_KEY_HERE` with your actual credentials!

### Step 3: Run Installation

```python
!python install_colab.py
```

### Step 4: Start the Bot

```python
!python main.py
```

---

## 🎯 What Happens Automatically

### ✅ **Hardware Detection**

```
🎮 GPU Detected: Tesla T4 (15.0 GB VRAM)
✅ PyTorch CUDA Available: True
```

### ✅ **Smart Installation**

- Detects T4 GPU automatically
- Installs CUDA-enabled dependencies
- Downloads AI models (Real-ESRGAN, GFPGAN)
- Skips heavy downloads if `FAST_MODE=yes`

### ✅ **Auto-Configuration**

- If `.env` is missing → Creates one with defaults
- If keys are missing → Auto-adds them
- GPU detected → Enables AI enhancement
- No GPU → Falls back to CPU/FFmpeg

---

## 🔧 Troubleshooting

### Issue: "No module named 'scipy'"

**Solution:** The bot auto-installs dependencies. If it fails:

```python
!pip install -r requirements.txt
```

### Issue: "CUDA out of memory"

**Solution:** Lower the enhancement level:

```python
# In .env
ENHANCEMENT_LEVEL=medium  # or basic
```

### Issue: Bot not responding

**Solution:** Check if bot token is correct:

```python
!cat .env | grep TELEGRAM_BOT_TOKEN
```

---

## 📊 Performance Modes (Colab T4)

| Mode                       | Speed      | Quality    | VRAM Usage | Best For        |
| -------------------------- | ---------- | ---------- | ---------- | --------------- |
| `FAST_MODE=yes`            | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | 0 GB       | Quick tests     |
| `ENHANCEMENT_LEVEL=basic`  | ⚡⚡⚡⚡   | ⭐⭐⭐     | 0 GB       | Fast processing |
| `ENHANCEMENT_LEVEL=medium` | ⚡⚡⚡     | ⭐⭐⭐⭐   | 4-6 GB     | Balanced        |
| `ENHANCEMENT_LEVEL=high`   | ⚡⚡       | ⭐⭐⭐⭐⭐ | 6-8 GB     | Best quality    |
| `ENHANCEMENT_LEVEL=ultra`  | ⚡         | ⭐⭐⭐⭐⭐ | 8-12 GB    | Maximum quality |

**Recommended for T4 (15GB VRAM):** `high` or `ultra`

---

## 🎬 Example Workflow

```python
# 1. Clone and setup
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME

# 2. Create .env (copy from above)
%%writefile .env
TELEGRAM_BOT_TOKEN=8359377646:AAH_rCZK6YUqmq3AyFSOo0ce8Fa5Nwh8fpc
GEMINI_API_KEY=AIzaSyBNH82hC6BdVXBXizysEGyq7K-q34n1Snw
FAST_MODE=no
ENHANCEMENT_LEVEL=high

# 3. Install
!python install_colab.py

# 4. Run
!python main.py
```

---

## 🚨 Important Notes

1. **`.env` file is NOT in GitHub** (it's in `.gitignore` for security)
   - You MUST create it manually in Colab (Step 2 above)
2. **Colab sessions are temporary**
   - Re-run setup each time you restart Colab
   - Your `.env` will be lost when session ends
3. **GPU Runtime Required**
   - Go to: Runtime → Change runtime type → T4 GPU
4. **Keep Colab Active**
   - Colab disconnects after ~90 minutes of inactivity
   - Use browser extensions to keep it alive

---

## 🎉 Expected Output

```
🔧 Starting Smart Startup Checks...
🎮 GPU Detected: Tesla T4 (15.0 GB VRAM)
✅ PyTorch CUDA Available: True
⚡ FAST_MODE: False (Source: no)
📦 Installing full dependencies (AI mode)...
🔧 Checking AI Models & Tools...
✅ Startup checks complete.

==================================================
🖥️  COMPUTE MODE SELECTION
==================================================
Auto-selected: GPU (Tesla T4, 15.0 GB VRAM)

🤖 Bot started successfully!
Send /start to begin
```

---

## 💡 Pro Tips

1. **Test with FAST_MODE first** to ensure everything works
2. **Then switch to AI mode** for production
3. **Monitor VRAM usage** with `!nvidia-smi`
4. **Save outputs to Google Drive** to persist files:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
