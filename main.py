from dotenv import load_dotenv
load_dotenv()

import os
import glob
import logging
import asyncio
import shutil
import sys
import re
import time
import subprocess
import csv
import json
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# Constants
ALLOWED_DOMAINS = ["instagram.com", "youtube.com", "youtu.be"]

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN not found in .env! Exiting.")
    sys.exit(1)

# Global State
user_sessions = {}
COMPILATION_BATCH_SIZE = int(os.getenv("COMPILATION_BATCH_SIZE", "5"))

# ==================== AUTO-INSTALL & SETUP ====================

# ==================== AUTO-INSTALL & SETUP ====================

def detect_hardware_capabilities():
    """
    Detect hardware capabilities for smart auto-selection.
    Returns: dict with 'has_gpu', 'gpu_name', 'vram_gb', 'cuda_available'
    """
    hardware_info = {
        'has_gpu': False,
        'gpu_name': 'CPU',
        'vram_gb': 0,
        'cuda_available': False
    }
    
    try:
        # Try to detect NVIDIA GPU without importing torch
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(',')
            hardware_info['has_gpu'] = True
            hardware_info['gpu_name'] = gpu_info[0].strip()
            hardware_info['vram_gb'] = int(gpu_info[1].strip().split()[0]) / 1024
            logger.info(f"🎮 GPU Detected: {hardware_info['gpu_name']} ({hardware_info['vram_gb']:.1f} GB VRAM)")
    except:
        logger.info("ℹ️ No NVIDIA GPU detected via nvidia-smi.")
    
    # Check PyTorch CUDA availability if GPU detected or just to be sure
    try:
        import torch
        if torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['has_gpu'] = True # Confirm GPU presence
            if hardware_info['gpu_name'] == 'CPU':
                hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
        
    return hardware_info

def resolve_compute_mode():
    """
    Resolve the final compute mode based on CPU_MODE, GPU_MODE settings and Hardware.
    Returns: 'gpu' or 'cpu'
    """
    cpu_mode = os.getenv("CPU_MODE", "auto").lower()
    gpu_mode = os.getenv("GPU_MODE", "auto").lower()
    
    # 1. Forced Modes
    if cpu_mode == "on":
        logger.info("🖥️ CPU_MODE is ON. Forcing CPU.")
        return "cpu"
    
    if gpu_mode == "on":
        logger.info("🎮 GPU_MODE is ON. Forcing GPU.")
        return "gpu"
        
    # 2. Auto Logic
    hardware = detect_hardware_capabilities()
    
    if gpu_mode == "auto":
        if hardware['cuda_available']:
            logger.info(f"🤖 GPU_MODE=auto: CUDA detected ({hardware['gpu_name']}). Selecting GPU.")
            return "gpu"
        elif hardware['has_gpu']:
             logger.info(f"🤖 GPU_MODE=auto: GPU detected but CUDA not ready. Falling back to CPU.")
             return "cpu"
        else:
            logger.info("🤖 GPU_MODE=auto: No GPU detected. Selecting CPU.")
            return "cpu"
            
    # Default fallback
    return "cpu"

def check_and_update_env():
    """
    Auto-updates .env file with missing keys and smart defaults.
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        logger.warning("⚠️ .env file not found. Creating template...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("""# ==================== CORE SETTINGS ====================
# REQUIRED: Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE

# REQUIRED: Get your API key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# ==================== PERFORMANCE ====================
# Modes: auto, on, off
CPU_MODE=auto
GPU_MODE=auto
REENCODE_PRESET=fast
REENCODE_CRF=25

# ==================== ENHANCEMENT ====================
ENHANCEMENT_LEVEL=medium
TARGET_RESOLUTION=1080:1920

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
DEFAULT_HASHTAGS_SHORTS=#shorts #viral #trending
DEFAULT_HASHTAGS_COMPILATION=#compilation #funny #viral

# ==================== TRANSITIONS ====================
TRANSITION_DURATION=0.5
TRANSITION_INTERVAL=5
""")
        logger.info("✅ Created .env template. Please update TELEGRAM_BOT_TOKEN and GEMINI_API_KEY!")
        
    # Load current env content
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    updates = []
    
    # Define required keys and defaults
    required_keys = {
        "CPU_MODE": "auto",
        "GPU_MODE": "auto",
        "ENHANCEMENT_LEVEL": "medium",
        "TRANSITION_INTERVAL": "5",
        "TRANSITION_DURATION": "0.5",
        "FORCE_AUDIO_REMIX": "yes",
        "ADD_TEXT_OVERLAY": "yes",
        "ADD_SPEED_RAMPING": "yes"
    }
    
    for key, default in required_keys.items():
        if key not in os.environ and f"{key}=" not in content:
            logger.info(f"➕ Auto-adding missing key: {key}={default}")
            updates.append(f"\n# Auto-added by Smart Installer\n{key}={default}")
            os.environ[key] = default 
            
    if updates:
        with open(env_path, "a", encoding="utf-8") as f:
            for line in updates:
                f.write(line)
        logger.info("✅ .env file updated with missing keys.")

def ensure_requirements_and_tools():
    """
    Smart Installation System based on CPU/GPU modes.
    """
    logger.info("🔧 Starting Smart Startup Checks...")
    
    # 0. Auto-Update .env
    check_and_update_env()
    
    # 1. Resolve Compute Mode
    mode = resolve_compute_mode()
    os.environ["COMPUTE_MODE"] = mode # Set for other scripts
    
    logger.info(f"⚡ FINAL COMPUTE MODE: {mode.upper()}")
    
    # 2. Smart Dependency Installation
    if mode == "cpu":
        logger.info("⚡ CPU Mode selected. Installing lightweight dependencies only...")
        lightweight_deps = ["python-telegram-bot", "python-dotenv", "requests", "google-generativeai", "yt-dlp"]
        try:
            for dep in lightweight_deps:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
    else:
        # GPU Mode
        logger.info("🎮 GPU Mode selected. Installing full AI dependencies...")
        
        # 1. Install Core Requirements first
        if os.path.exists("requirements.txt"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error(f"❌ Failed to install core requirements: {e}")

        # 2. Auto-Install AI Libraries (Torch, RealESRGAN, YOLO)
        ai_libs = [
            "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2",
            "realesrgan==0.3.0", "gfpgan==1.3.8", "basicsr==1.4.2", 
            "facexlib==0.3.0", "ultralytics"
        ]
        
        logger.info("📦 Checking/Installing AI Libraries...")
        for lib in ai_libs:
            try:
                # Check if installed (simple check)
                pkg_name = lib.split("==")[0]
                subprocess.check_call([sys.executable, "-m", "pip", "show", pkg_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                logger.info(f"   ⬇️ Installing {lib}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "-q"], 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to install {lib}: {e}")

        # Install AI tools
        try:
            logger.info("🔧 Checking AI Models & Tools...")
            subprocess.check_call([sys.executable, "tools-install.py"])
        except Exception as e:
            logger.error(f"❌ Tools installation failed: {e}")
        
    logger.info("✅ Startup checks complete.")

# Run setup BEFORE imports that might need them
# Run setup BEFORE imports that might need them
ensure_requirements_and_tools()

# Late Import of Telegram (after install)
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.error import NetworkError, TimedOut
except ImportError:
    logger.error("❌ python-telegram-bot not found! Install failed.")
    sys.exit(1)

# Conditional imports
compute_mode = os.environ.get("COMPUTE_MODE", "cpu")

try:
    import downloader
    import uploader
    from compiler import compile_with_transitions, compile_batch_with_transitions
    from router import run_enhancement
    
    # Only import audio_processing if we have full dependencies (GPU mode)
    if compute_mode == "gpu":
        try:
            import audio_processing
        except ImportError:
            logger.warning("⚠️ audio_processing not found (likely CPU mode).")
            audio_processing = None
    else:
        logger.info("ℹ️ Skipping audio_processing import (CPU mode)")
        audio_processing = None
        
except ImportError as e:
    logger.error(f"Critical Import Error: {e}")
    sys.exit(1)

# ==================== UTILS ====================

UPLOAD_LOG = "upload_log.csv"

def _ensure_log_header():
    if not os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "file_path", "yt_link", "title"])

def log_video(file_path: str, yt_link: str, title: str):
    _ensure_log_header()
    with open(UPLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), file_path, yt_link, title])

def total_uploads() -> int:
    if not os.path.exists(UPLOAD_LOG):
        return 0
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        return max(0, len(rows) - 1)

def last_n_filepaths(n: int) -> list:
    """Get the last N video file paths from the upload log, filtered by recency."""
    if not os.path.exists(UPLOAD_LOG):
        return []
    
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter by timestamp - only videos from last 24 hours
    from datetime import datetime, timedelta
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    recent_rows = []
    for r in rows:
        try:
            timestamp_str = r.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_rows.append(r)
        except:
            # If timestamp parsing fails, skip this row
            continue
    
    # Get last N from recent rows
    subset = recent_rows[-n:]
    paths = [r.get("file_path") for r in subset if r.get("file_path")]
    
    # Return only paths that exist
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    
    logger.info(f"📊 Found {len(valid_paths)} recent videos for compilation (last 24h)")
    return valid_paths

async def safe_reply(update: Update, text: str):
    for _ in range(3):
        try:
            if update.message:
                await update.message.reply_text(text)
            return
        except (NetworkError, TimedOut) as e:
            logger.warning("🛑 Reply failed: %s. Retrying...", e)
            await asyncio.sleep(2)
    logger.error("❌ Failed to send message after retries.")

async def safe_video_reply(update: Update, video_path: str, caption: str = None):
    """
    Robustly reply with a video, handling timeouts and retries.
    """
    for attempt in range(1, 4):
        try:
            if update.message:
                # read_timeout/write_timeout kwargs are supported in send_video (which reply_video wraps)
                # We set a very high timeout for large file uploads
                await update.message.reply_video(
                    video_path, 
                    caption=caption, 
                    read_timeout=600, 
                    write_timeout=600,
                    connect_timeout=60,
                    pool_timeout=60
                )
            return
        except (NetworkError, TimedOut) as e:
            logger.warning(f"🛑 Video reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"❌ Video reply error: {e}")
            break
            
    logger.error("❌ Failed to send video after retries.")
    await safe_reply(update, "❌ Failed to send video due to network timeout.")

def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(allowed in domain for allowed in ALLOWED_DOMAINS)
    except: return False

def _sanitize_title(title: str) -> str:
    # Allow spaces but remove other special characters
    clean = re.sub(r'[^\w\s-]', '', title)
    # clean = clean.replace(' ', '_')  <-- REMOVED: Keep spaces for YouTube title
    return clean[:100]  # Increased limit slightly for better titles

def _get_hashtags(text: str) -> str:
    link_count = len(re.findall(r'https?://', text))
    if link_count > 1:
        return os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").strip()
    return os.getenv("DEFAULT_HASHTAGS_SHORTS", "").strip()



# ==================== COMPILATION LOGIC ====================

async def maybe_compile_and_upload(update: Update):
    count = total_uploads()
    n = COMPILATION_BATCH_SIZE
    if n <= 0 or count == 0 or count % n != 0:
        return

    await safe_reply(update, f"⏳ Creating compilation of last {n} shorts...📦")
    files = last_n_filepaths(n)
    if len(files) < n:
        await safe_reply(update, "⚠️ Not enough local files to compile. Skipping.")
        return

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = f"compilation_{n}_{stamp}.mp4"
    await safe_reply(update, f"🔨 Merging {len(files)} videos now...🛸")

    try:
        await safe_reply(update, "✨ Running full AI pipeline for batch compilation…")

        # --- Single Stage: Batch Compile with Transitions ---
        # This replaces the old 2-stage process (raw merge -> enhance)
        # Now we normalize -> transition -> merge -> remix -> assemble in one go
        
        output_filename = f"compilation_{n}_{stamp}.mp4"
        
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            files,
            output_filename
        )
        
        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Failed to create compilation.")
            return

        # Check if we should send to YouTube or Telegram
        send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
        
        if not send_to_youtube:
            await safe_reply(update, "📤 Sending compilation for testing...")
            if os.path.getsize(merged) < 50 * 1024 * 1024:
                await safe_video_reply(update, merged)
            else:
                await safe_reply(update, "⚠️ Compilation too large for Telegram.")
            return

        comp_title = f"🎬 {n} Videos Compilation #{count // n}"  # Changed from "Shorts" to "Videos"
        
        # Use compilation hashtags WITHOUT #Shorts to ensure it's uploaded as regular video
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").replace("#Shorts", "").replace("#shorts", "").strip()
        
        comp_link = await uploader.upload_to_youtube(merged, comp_hashtags, comp_title)

        if comp_link:
            await safe_reply(update, f"🎉 Compilation uploaded!\n🔗 {comp_link}")
            log_video(merged, comp_link, comp_title)
        else:
            await safe_reply(update, "❌ Failed to upload compilation.")

    except Exception as e:
        logger.exception("Compilation/upload failed: %s", e)
        await safe_reply(update, f"❌ Compilation failed: {e}")

async def compile_last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compiles the last N downloaded videos from the downloads/ folder.
    Usage: /compile_last <number> (default 6)
    """
    try:
        # 1. Parse arguments
        n = 6
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                await safe_reply(update, "⚠️ Invalid number. Using default: 6")
        
        if n <= 1:
            await safe_reply(update, "⚠️ Please specify at least 2 videos.")
            return

        # Source from Processed Shorts
        source_dir = "Processed Shorts"
        if not os.path.exists(source_dir):
             await safe_reply(update, f"❌ Directory '{source_dir}' not found.")
             return

        all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
        files = [f for f in all_files if not os.path.basename(f).startswith("compile_")]
        
        if not files:
            await safe_reply(update, f"❌ No processed videos found in '{source_dir}' folder.")
            return

        # 3. Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Take top N
        selected_files = files[:n]
        
        if len(selected_files) < 2:
            await safe_reply(update, f"⚠️ Found {len(selected_files)} videos, but need at least 2 to compile.")
            return

        # Log selected files for user confirmation
        msg = f"✅ Found {len(selected_files)} videos:\n"
        for f in selected_files:
            msg += f"- {os.path.basename(f)}\n"
        await safe_reply(update, msg)

        # 4. Compile
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compile_last_{n}_{stamp}.mp4"
        
        await safe_reply(update, "🚀 Starting batch compilation with transitions...")
        
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            selected_files,
            output_filename
        )

        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Compilation failed (check logs).")
            return

        # 5. Send Result
        await safe_reply(update, "📤 Sending compiled video...")
        if os.path.getsize(merged) < 50 * 1024 * 1024:
            await safe_video_reply(update, merged, caption=f"🎬 Last {len(selected_files)} Videos Compilation")
        else:
            await safe_reply(update, "⚠️ Video too large for Telegram, but saved locally.")
            
        logger.info(f"✅ /compile_last finished: {merged}")

    except Exception as e:
        logger.exception(f"/compile_last failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

# ==================== HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "❓ Please send an Instagram reel or YouTube link to begin.")

async def getbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, f"Current compilation batch size: {COMPILATION_BATCH_SIZE}")

async def setbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global COMPILATION_BATCH_SIZE
    try:
        if not context.args:
            await safe_reply(update, "Usage: /setbatch <number>")
            return
        n = int(context.args[0])
        if n <= 0:
            await safe_reply(update, "Please provide a positive integer.")
            return
        COMPILATION_BATCH_SIZE = n
        await safe_reply(update, f"✅ Compilation batch size set to {n}.")
    except Exception:
        await safe_reply(update, "Usage: /setbatch <number>")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    load_dotenv(override=True)
    send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
    
    text = update.message.text.strip()
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    state = session.get('state')
    
    # Case 1: Link
    if _validate_url(text):
        hashtags = _get_hashtags(text)
        user_sessions[user_id] = {'state': 'WAITING_FOR_TITLE', 'url': text, 'hashtags': hashtags}
        await safe_reply(update, "✅ Got the link!")
        await safe_reply(update, f"📌 Hashtags:\n{hashtags}")
        await safe_reply(update, "✏️ Now send the title.")
        return

    # Case 2: Title
    if state == 'WAITING_FOR_TITLE':
        url = session.get('url')
        hashtags = session.get('hashtags')
        title = _sanitize_title(text)
        
        if not title:
            await safe_reply(update, "❌ Invalid title.")
            return
            
        user_sessions.pop(user_id, None)
        
        try:
            await safe_reply(update, "📥 Downloading...")
            video_path = await asyncio.to_thread(downloader.download_video, url, custom_title=title)
            
            if not video_path or not os.path.exists(video_path):
                await safe_reply(update, "❌ Download failed.")
                return

            await safe_reply(update, "🎧 Processing audio...")
            await safe_reply(update, "🧑‍🎨 Enhancing...")
            await safe_reply(update, "🎬 Finalizing output (Transitions Engine)...")
            
            final_path, wm_context = await asyncio.to_thread(compile_with_transitions, Path(video_path), title)
            
            if not final_path or not os.path.exists(final_path):
                await safe_reply(update, "❌ Processing failed.")
                return
                
            final_str = str(final_path)
            
            # QA: Send for Review
            user_sessions[user_id] = {
                'state': 'WAITING_FOR_APPROVAL',
                'final_path': final_str,
                'title': title,
                'hashtags': hashtags,
                'watermark_context': wm_context # Store for feedback
            }
            
            await safe_reply(update, "✅ Video processed! Sending preview...")
            
            caption = f"✨ {title}\n\nReply /approve to upload or /reject to discard.\n\n(Watermark detected - please verify removal- yes/no)"
            
            if os.path.getsize(final_str) < 50 * 1024 * 1024:
                await safe_video_reply(update, final_str, caption=caption)
            else:
                await safe_reply(update, "⚠️ Video too large for Telegram preview.\nReply /approve to upload blindly or /reject.")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            await safe_reply(update, "❌ Error occurred.")
        return

    # Case 3: Approval
    if state == 'WAITING_FOR_APPROVAL':
        if text.lower() in ['approve', '/approve']:
            await approve_upload(update, context)
        elif text.lower() in ['yes', 'y']:
            await verify_watermark(update, context, is_positive=True)
        elif text.lower() in ['no', 'n']:
            await verify_watermark(update, context, is_positive=False)
        elif text.lower() in ['reject', '/reject']:
            await reject_upload(update, context)
        else:
            await safe_reply(update, "⚠️ Options:\n• 'yes'/'no' - Verify watermark removal (Training Data)\n• '/approve' - Upload to YouTube\n• '/reject' - Discard Video")
        return

async def verify_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE, is_positive: bool = True):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get('state') != 'WAITING_FOR_APPROVAL':
        await safe_reply(update, "⚠️ No video waiting for approval.")
        return

    wm_context = session.get('watermark_context')
    
    if wm_context:
        coords = wm_context.get('coords')
        
        if coords:
            # Case 1: System detected something (Coords exist)
            try:
                import watermark_auto
                watermark_auto.confirm_learning(wm_context, is_positive=is_positive)
                
                if is_positive:
                    await safe_reply(update, "🧠 Watermark detection confirmed (Learned POSITIVE).")
                else:
                    await safe_reply(update, "🧠 Watermark detection rejected (Learned NEGATIVE).")
                    
                # Clear wm_context so we don't learn again
                session['watermark_context'] = None
            except Exception as e:
                logger.warning(f"Failed to confirm learning: {e}")
        else:
            # Case 2: System detected NOTHING (No Coords)
            if is_positive:
                # User says "Yes" -> Correctly detected nothing
                await safe_reply(update, "✅ Feedback recorded: Correctly identified NO watermark.")
            else:
                # User says "No" -> Missed watermark
                await safe_reply(update, "⚠️ Feedback recorded: Missed watermark. (Cannot learn without coordinates).")
            
            session['watermark_context'] = None
            
    else:
        await safe_reply(update, "ℹ️ Feedback noted (No watermark context available).")
        
    await safe_reply(update, "✅ Feedback recorded.\nReply /approve to upload or /reject to discard.")

async def approve_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get('state') != 'WAITING_FOR_APPROVAL':
        await safe_reply(update, "⚠️ No video waiting for approval.")
        return

    final_path = session.get('final_path')
    title = session.get('title')
    hashtags = session.get('hashtags')
    
    if not final_path or not os.path.exists(final_path):
        await safe_reply(update, "❌ Video file expired or missing.")
        user_sessions.pop(user_id, None)
        return

    await safe_reply(update, "📤 Uploading to YouTube...")
    try:
        link = await uploader.upload_to_youtube(final_path, title=title, hashtags=hashtags)
        if link:
            await safe_reply(update, f"🎉 Uploaded successfully!\n🔗 {link}")
            log_video(final_path, link, title)
            
            # Check for compilation trigger
            await maybe_compile_and_upload(update)
        else:
            await safe_reply(update, "❌ Upload failed.")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        await safe_reply(update, f"❌ Upload error: {e}")
        
    # Clear session
    user_sessions.pop(user_id, None)

async def reject_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get('state') == 'WAITING_FOR_APPROVAL':
        user_sessions.pop(user_id, None)
        await safe_reply(update, "🗑️ Video discarded.")
    else:
        await safe_reply(update, "⚠️ Nothing to reject.")

def _bootstrap():
    if not shutil.which("ffmpeg"):
        logger.error("❌ FFmpeg not found.")
        sys.exit(1)

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        sys.exit(1)
    _bootstrap()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).read_timeout(600).write_timeout(600).connect_timeout(60).pool_timeout(60).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("approve", approve_upload))
    app.add_handler(CommandHandler("reject", reject_upload))
    app.add_handler(CommandHandler("getbatch", getbatch))
    app.add_handler(CommandHandler("setbatch", setbatch))
    app.add_handler(CommandHandler("compile_last", compile_last))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    logger.info("🤖 Bot is running...")
    app.run_polling()
