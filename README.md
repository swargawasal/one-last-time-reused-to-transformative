# 🎬 YouTube Automation Bot - Self-Learning AI Video Enhancement

**Transform reused content into viral-ready videos with Hybrid Vision AI, Self-Learning Watermark Detection, and Smart Audio Remixing.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## ✨ Key Features

### 🧠 **Self-Learning Watermark System (NEW)**

- **Hybrid Vision**: Combines **Google Gemini Vision** (Context) + **OpenCV ORB** (Precision) + **Machine Learning** (Validation).
- **Dual-Memory Learning**: Maintains separate banks for **Positive** (Valid) and **Negative** (False Positive) templates.
- **User Feedback Loop**: Learns from your Telegram commands:
  - `/approve`: "This was a correct detection." (Learns Positive)
  - `/reject`: "This was a mistake." (Learns Negative)
- **Data-Driven ML**: Automatically trains a Random Forest classifier to predict watermark validity based on texture, geometry, and match confidence.

### 🎨 **Transformative Content Engine**

- **Gemini AI Captions**: Generates viral-style captions, titles, and hashtags using Gemini 1.5 Flash.
- **Smart Cropping**: Intelligent content-aware cropping for 9:16 vertical format.
- **Dynamic Text Overlays**: Professional-grade text rendering with shadows and borders.
- **Cinematic Color Grading**: Applies "Dark", "Vibrant", or "Cinematic" LUTs.
- **Speed Ramping**: Dynamic speed variations to avoid copyright matching.

### 🎵 **Advanced Audio Studio**

- **Heavy Remixing**: Completely transforms audio structure using beat-aware slicing.
- **Auto-Generated Music**: Creates unique, copyright-free background music on the fly.
- **Continuous Mixes**: Stitches multiple tracks from your library for long compilations.
- **Voiceover Mixing**: Auto-ducks background music for AI voiceovers.

### 🚀 **Performance & Architecture**

- **Smart Hardware Detection**: Auto-switches between NVENC (GPU) and libx264 (CPU).
- **Fast Mode**: 10x faster processing for quick iterations.
- **Batch Compilation**: Merges multiple processed clips with smart transitions (Fade, Wipe, Zoom).

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended for GPU)**

1. **Clone the repository:**

   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   %cd YOUR_REPO
   ```

2. **Create `.env` file:**

   ```python
   %%writefile .env
   TELEGRAM_BOT_TOKEN=your_token_here
   GEMINI_API_KEY=your_key_here
   FAST_MODE=no
   ENHANCEMENT_LEVEL=high
   ```

3. **Run installation:**

   ```python
   !python install_colab.py
   ```

4. **Start the bot:**
   ```python
   !python main.py
   ```

---

## ⚙️ Configuration (`.env`)

### **Core Settings**

```ini
TELEGRAM_BOT_TOKEN=your_token
GEMINI_API_KEY=your_key
```

### **Watermark & Vision**

```ini
WATERMARK_DETECTION=yes
WATERMARK_REMOVE=yes
WATERMARK_DETECT_LIST=tiktok,reels,watermark,@  # Keywords for Gemini
WATERMARK_MAX_AREA_PERCENT=20                   # Safety check
```

### **Audio & Transformative**

```ini
ENABLE_HEAVY_REMIX_SHORTS=yes       # Aggressive remixing for Shorts
ENABLE_AUTO_MUSIC_GEN=yes           # Generate unique music if needed
AI_CAPTIONS=yes                     # Use Gemini for captions
ADD_COLOR_GRADING=yes
ADD_SPEED_RAMPING=yes
```

---

## 🎬 Usage

### **Telegram Commands**

- `/start` - Start the bot.
- **Send Video/Link** - Process a single video.
- **Reply `/approve`** - Confirm watermark removal (Learns **Positive**).
- **Reply `/reject`** - Discard result (Learns **Negative** - False Positive).
- `/compile_last <N>` - Compile the last N processed videos into a long video.
- `/setbatch <N>` - Set auto-compilation threshold.

### **Workflow**

1. **Send Video**: Bot downloads and analyzes.
2. **Detection**: Gemini finds watermark -> OpenCV verifies.
3. **Review**: Bot sends preview with "Watermark detected - verify removal".
4. **Feedback**: You reply `/approve` or `/reject`.
5. **Learning**: Bot updates its database (`watermark_templates/` and `watermark_dataset.csv`).
6. **Upload**: If approved, bot uploads to YouTube (if configured).

---

## 📁 Project Structure

```
├── main.py                 # Bot entry point & Telegram handlers
├── compiler.py             # Core video processing pipeline
├── watermark_auto.py       # Hybrid Vision orchestration
├── opencv_watermark.py     # ORB + ML Learning Engine
├── gemini_captions.py      # AI Captioning & Vision
├── audio_processing.py     # Remixing & Music Generation
├── ai_engine.py            # Upscaling (Real-ESRGAN/GFPGAN)
├── watermark_templates/    # Learned templates (Positive/Negative)
├── watermark_dataset.csv   # ML Training Data
└── requirements.txt        # Dependencies
```

---

## 📊 Performance Benchmarks

| Hardware       | Mode       | Video (23s) | Processing Time |
| -------------- | ---------- | ----------- | --------------- |
| CPU (i7)       | FAST_MODE  | 23s         | ~30s            |
| GPU (T4, 15GB) | AI (high)  | 23s         | ~30s ✅         |
| GPU (T4, 15GB) | AI (ultra) | 23s         | ~1 min          |

---

**Made with ❤️ for content creators**
