# ============================================================
# GOOGLE COLAB INSTALLATION SCRIPT - ROBUST & UNIFIED
# Run this FIRST before running main.py
# ============================================================

print("🚀 Installing YouTube Automation Bot for Google Colab...")
print("=" * 60)

import subprocess
import sys
import os

def run_cmd(cmd, desc):
    print(f"\n📦 {desc}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print("✅ Done")
    except subprocess.CalledProcessError as e:
try:
    import site
    site_packages = site.getsitepackages()[0]
    degradations_file = os.path.join(site_packages, 'basicsr', 'data', 'degradations.py')
    
    if os.path.exists(degradations_file):
        with open(degradations_file, 'r') as f:
            content = f.read()
        
        # Fix the import
        new_content = content.replace(
            'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
            'from torchvision.transforms.functional import rgb_to_grayscale'
        )
        
        if content != new_content:
            with open(degradations_file, 'w') as f:
                f.write(new_content)
            print("✅ basicsr patched successfully")
        else:
            print("ℹ️ basicsr already patched or different version")
    else:
        print("⚠️ basicsr not found, skipping patch (might be installed later)")
except Exception as e:
    print(f"⚠️ Patching failed: {e}")

# Step 5: Install Heavy AI Tools
print("\n📦 Installing Heavy AI Tools (RealESRGAN/GFPGAN)...")
try:
    subprocess.run([sys.executable, "tools-install.py"], check=True)
except Exception as e:
    print(f"⚠️ Tools install warning: {e}")

# Step 6: Verify installation
print("\n📦 Verifying installation...")
try:
    import numpy as np
    import cv2
    import torch
    
    print(f"  ✅ NumPy: {np.__version__}")
    print(f"  ✅ OpenCV: {cv2.__version__}")
    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✅ GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("  ⚠️ No GPU detected - AI enhancement will be disabled")
except ImportError as e:
    print(f"❌ Verification failed: {e}")

# Step 7: Setup environment variables
print("\n📦 Setting up environment...")
print("Please configure your .env file with:")
print("  - TELEGRAM_BOT_TOKEN")
print("  - GEMINI_API_KEY")
print("\nYou can use Colab Secrets or create .env manually")

print("\n" + "=" * 60)
print("✅ Installation complete!")
print("⚠️ IMPORTANT: Ignore NumPy/scipy dependency warnings")
print("   (They won't affect the bot's functionality)")
print("\nRun: !python main.py")
print("=" * 60)
