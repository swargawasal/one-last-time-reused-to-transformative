"""
CPU Fast Processing Module
FFmpeg-based fast processing for upscaling and enhancement.
"""

import os
import subprocess
import logging

logger = logging.getLogger("cpu_fast")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def _run_ffmpeg(cmd):
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e}")
        return False

def fast_upscale(input_path, output_path, scale=2):
    """
    Fast CPU upscaling using Lanczos (FFmpeg).
    """
    logger.info(f"⚡ CPU Fast Upscale (x{scale}) using FFmpeg...")
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos"
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    return _run_ffmpeg(cmd)

def smart_reframe_cpu(input_path, output_path, target_res="1080:1920"):
    """
    CPU-based crop/pad to target resolution.
    """
    logger.info(f"⚡ CPU Smart Reframe to {target_res}...")
    w, h = map(int, target_res.split(":"))
    
    # Simple center crop/pad strategy for CPU
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    return _run_ffmpeg(cmd)

def apply_fallback_enhancement(input_path, output_path):
    """
    MAXIMUM FFmpeg Enhancement Patch.
    Applies 2x sharpening, cinematic contrast, normalized brightness,
    boosted saturation, temporal light grain, and lanczos upscale.
    """
    logger.info("⚡ Applying MAXIMUM FFmpeg Enhancement (Fallback Mode)...")
    
    # New Filter Chain (MAXIMUM QUALITY):
    # 1. unsharp=7:7:1.2 -> 2x sharpening
    # 2. eq=contrast=1.18:brightness=0.03:saturation=1.25 -> Cinematic grading
    # 3. noise=alls=3:allf=t -> Temporal light grain (prevents soft video)
    # 4. scale=1080:1920:flags=lanczos -> Best CPU upscale
    vf = "unsharp=7:7:1.2,eq=contrast=1.18:brightness=0.03:saturation=1.25,noise=alls=3:allf=t,scale=1080:1920:flags=lanczos"
        
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    return _run_ffmpeg(cmd)
