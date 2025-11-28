"""
Gemini AI Video Orchestrator Module
Analyzes video frames and outputs JSON instructions for FFmpeg/OpenCV.
Does NOT generate pixels directly.
"""

import os
import cv2
import base64
import logging
import time
import json
import re
import gc
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("gemini_orchestrator")

# Try to import Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("⚠️ google-generativeai not installed. Gemini orchestrator disabled.")

# Configuration
ENABLE_GEMINI_ENHANCE = os.getenv("ENABLE_GEMINI_ENHANCE", "yes").lower() == "yes"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "15"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_FAILOVER = os.getenv("GEMINI_FAILOVER", "yes").lower() == "yes"

# Watermark & Crop Config
ENABLE_GEMINI_WATERMARK_DETECT = os.getenv("ENABLE_GEMINI_WATERMARK_DETECT", "yes").lower() == "yes"
WATERMARK_MAX_AREA_PERCENT = int(os.getenv("WATERMARK_MAX_AREA_PERCENT", "15"))
ENABLE_SMART_CROP = os.getenv("ENABLE_SMART_CROP", "no").lower() == "yes"
FORCE_NO_CROP = os.getenv("FORCE_NO_CROP", "yes").lower() == "yes"

gemini_client = None

def init_gemini(api_key: str) -> bool:
    global gemini_client
    if not HAS_GEMINI or not api_key: return False
    
    try:
        genai.configure(api_key=api_key)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"✅ Gemini Orchestrator initialized: {GEMINI_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")
        return False

def frame_to_base64(frame: np.ndarray) -> Optional[str]:
    try:
        # Resize for analysis speed (max 1024px width)
        h, w = frame.shape[:2]
        if w > 1024:
            scale = 1024 / w
            frame = cv2.resize(frame, (1024, int(h * scale)))
            
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success: return None
        return base64.b64encode(buffer).decode('utf-8')
    except:
        return None

def get_orchestrator_prompt(dynamic_keywords: str = "") -> str:
    return f"""
You are an Elite Video Automation Orchestrator.
ANALYZE this video frame and output JSON instructions.

==========================================================================
PART A — DYNAMIC INSTAGRAM WATERMARK DETECTION
==========================================================================

You will receive a dynamic list of watermark keywords extracted from:
- Instagram uploader username
- Variations: "@username", "username", "user name", "usernameofficial"
- Caption hashtags (#viralbhayani #instantbollywood etc.)
- Caption text
- Cleaned words (remove emojis, special characters)

This list is provided as:

{dynamic_keywords}

Your task:
- Detect ANY text/logo in the video frame matching ANY keyword above
- Case-insensitive
- Partial matches allowed ("viral" matches "viralbhayani")
- Transparent text MUST be detected
- Small corner logos MUST be detected
- Username watermarks MUST be detected
- Only detect if area <= {int(os.getenv("WATERMARK_MAX_AREA_PERCENT", "15"))}%

Return ONLY JSON:

{{
  "watermark_detected": true/false,

{{
  "summary": {{
    "do_enhancement": true/false,
    "do_watermark_remove": true/false,
    "do_watermark_replace": true/false,
    "do_crop": true/false
  }}
}}

==========================================================================
IMPORTANT RULES
==========================================================================

- YOU NEVER generate videos.
- YOU NEVER modify pixels.
- YOU ONLY return JSON instructions.
- FFmpeg + OpenCV handle enhancement + delogo.
- ALWAYS return valid JSON.
"""

def clean_json_response(text: str) -> str:
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def analyze_frame(frame: np.ndarray, dynamic_keywords: str = "") -> Dict[str, Any]:
    """
    Analyze a single frame and return instructions.
    """
    global gemini_client
    if not gemini_client: return None
    
    try:
        b64_frame = frame_to_base64(frame)
        if not b64_frame: return None
        
        prompt = get_orchestrator_prompt(dynamic_keywords)
        
        # Define safety settings to prevent blocking (List format for compatibility)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Call Gemini
        response = gemini_client.generate_content(
            contents=[
                {'mime_type': 'image/jpeg', 'data': b64_frame},
                prompt
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Low temp for deterministic JSON
                max_output_tokens=1024
            ),
            safety_settings=safety_settings
        )
        
        # Parse JSON
        try:
            json_str = clean_json_response(response.text)
            instructions = json.loads(json_str)
            return instructions
        except ValueError:
            # response.text raises ValueError if blocked
            logger.warning(f"⚠️ Gemini blocked content (Safety). Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
            return None
        except Exception as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None

def run(input_video: str, output_video: str) -> str:
    """
    Orchestrator Entry Point.
    1. Analyzes representative frame.
    2. Generates FFmpeg command.
    3. Executes FFmpeg.
    """
    if not ENABLE_GEMINI_ENHANCE: return "GEMINI_FAIL"
    
    # Init
    if not gemini_client:
        if not init_gemini(os.getenv("GEMINI_API_KEY")):
            return "GEMINI_FAIL"
            
    try:
        logger.info("🤖 Gemini Orchestrator: Analyzing video...")
        
        # Extract middle frame for analysis
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return "GEMINI_FAIL"
        
        # Try to load metadata for dynamic keywords
        dynamic_keywords = ""
        try:
            meta_path = input_video.rsplit('.', 1)[0] + '.json'
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    
                keywords = []
                if meta.get('uploader'):
                    u = str(meta['uploader'])
                    keywords.append(u)
                    keywords.append(f"@{u}")
                    keywords.append(u.replace(" ", ""))
                    keywords.append(u.lower())
                
                if meta.get('caption'):
                    # Extract hashtags
                    tags = re.findall(r'#(\w+)', str(meta['caption']))
                    keywords.extend([f"#{t}" for t in tags])
                    keywords.extend(tags)
                    
                if meta.get('tags'):
                    keywords.extend([str(t) for t in meta['tags']])
                    
                # Deduplicate and clean
                unique_keywords = list(set([k for k in keywords if k]))
                dynamic_keywords = ", ".join(unique_keywords)
                logger.info(f"🔍 Dynamic Watermark Keywords: {dynamic_keywords[:100]}...")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load metadata for dynamic keywords: {e}")
        
        # Analyze
        instructions = analyze_frame(frame, dynamic_keywords)
        if not instructions: return "GEMINI_FAIL"
        
        logger.info(f"📋 Instructions received: {json.dumps(instructions, indent=2)}")
        
        # --- CONSTRUCT FFMPEG COMMAND ---
        filters = []
        
        # 1. Enhancement
        enh = instructions.get("enhancement", {})
        if enh.get("enhance", False):
            # Sharpness (unsharp)
            # Map 0.0-1.0 to unsharp params (luma_msize_x:luma_msize_y:luma_amount)
            sharp = float(enh.get("sharpness", 0))
            if sharp > 0:
                amount = 0.5 + (sharp * 1.5) # 0.5 to 2.0
                filters.append(f"unsharp=5:5:{amount:.2f}:5:5:0.0")
                
            # Color/Contrast (eq)
            cont = float(enh.get("contrast", 1.0))
            bright = float(enh.get("brightness", 0.0))
            sat = float(enh.get("saturation", 1.0))
            if cont != 1.0 or bright != 0.0 or sat != 1.0:
                filters.append(f"eq=contrast={cont:.2f}:brightness={bright:.2f}:saturation={sat:.2f}")
                
            # Denoise (hqdn3d)
            denoise = float(enh.get("denoise", 0))
            if denoise > 0:
                val = denoise * 10 # 0 to 10
                filters.append(f"hqdn3d={val:.1f}:{val:.1f}:6:6")
                
            # Upscale
            if enh.get("upscale") == "2x":
                filters.append("scale=iw*2:ih*2:flags=lanczos")
            else:
                filters.append("scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

        # 2. Watermark Removal (delogo)
        wm = instructions.get("watermark", {})
        safe = instructions.get("safety", {}).get("safe_to_remove", False)
        
        if wm.get("watermark_detected") and safe and not ENABLE_GEMINI_WATERMARK_DETECT:
             # Only if enabled in .env (user disabled it earlier)
             pass 
        elif wm.get("watermark_detected") and safe and ENABLE_GEMINI_WATERMARK_DETECT:
             coords = wm.get("coordinates", {})
             x, y, w, h = coords.get("x"), coords.get("y"), coords.get("w"), coords.get("h")
             if all(v is not None for v in [x,y,w,h]):
                 filters.append(f"delogo=x={x}:y={y}:w={w}:h={h}:show=0")
                 logger.info(f"   🛡️ Removing watermark at {x},{y} {w}x{h}")

        # 3. Crop (Optional)
        crop = instructions.get("crop", {})
        if crop.get("crop_required") and ENABLE_SMART_CROP and not FORCE_NO_CROP:
             b = crop.get("crop_box", {})
             filters.append(f"crop={b.get('w')}:{b.get('h')}:{b.get('x')}:{b.get('y')}")

        # Build Command
        vf_chain = ",".join(filters) if filters else "null"
        
        # If no filters, just copy (but we usually have scale)
        if not filters: vf_chain = "scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", vf_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_video
        ]
        
        logger.info(f"⚡ Executing FFmpeg: {vf_chain}")
        
        import subprocess
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return "SUCCESS"

    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        return "GEMINI_FAIL"
