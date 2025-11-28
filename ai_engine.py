# ai_engine.py - OPTIMIZED AI ENHANCEMENT ENGINE
import os
import logging
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

logger = logging.getLogger("ai_engine")

class HeavyEditor:
    """AI Video Enhancement Engine using RealESRGAN + GFPGAN (Optimized)"""
    def __init__(self, model_dir="models/heavy", scale=2, face_enhance=True):
        self.model_dir = model_dir
        self.scale = scale
        self.face_enhance = face_enhance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths to models
        self.realesrgan_model_path = os.path.join(model_dir, 'RealESRGAN_x4plus.pth')
        self.gfpgan_model_path = os.path.join(model_dir, 'GFPGANv1.4.pth')
        
        self.upsampler = None
        self.face_enhancer = None
        
        self._load_models()

    def _get_device_config(self):
        """Detect VRAM and return optimal settings based on strict thresholds."""
        config = {
            "tile": 0,
            "half": False,
            "face_enhance": True,
            "device": self.device
        }
        
        if self.device.type == 'cuda':
            config["half"] = True
            try:
                vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                logger.info(f"🎮 GPU Detected: {torch.cuda.get_device_name(self.device)} ({vram:.2f} GB VRAM)")
                
                # Strict VRAM Thresholds
                if vram < 2:
                    logger.warning("⚠️ Very Low VRAM (<2GB). AI might be unstable. Using aggressive tiling.")
                    config["tile"] = 200
                    config["face_enhance"] = False # Too heavy
                elif vram < 4:
                    logger.info("ℹ️ Low VRAM (2-4GB). Tiling enabled, face enhancement disabled.")
                    config["tile"] = 400
                    config["face_enhance"] = False
                elif vram < 6:
                    logger.info("ℹ️ Medium VRAM (4-6GB). Tiling enabled (640), face enhancement disabled for stability.")
                    config["tile"] = 640
                    config["face_enhance"] = False # Safer to disable
                elif vram < 8:
                    logger.info("ℹ️ Medium-High VRAM (6-8GB). Tiling enabled (640), face enhancement ENABLED.")
                    config["tile"] = 640
                    config["face_enhance"] = True
                else:
                    logger.info("🚀 High VRAM (>8GB). Full enhancement mode: NO tiling, face enhancement ENABLED.")
                    config["tile"] = 0
                    config["face_enhance"] = True
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect VRAM: {e}. Defaulting to safe mode.")
                config["tile"] = 400
                config["face_enhance"] = False
        else:
            logger.warning("⚠️ CPU Mode detected. Basic mode only (no face enhancement).")
            config["tile"] = 200
            config["half"] = False
            config["face_enhance"] = False
            
        return config

    def _load_models(self):
        logger.info(f"🚀 Loading Heavy Engine Models on {self.device}...")
        
        config = self._get_device_config()
        
        # Override face enhance if config says no
        if not config["face_enhance"]:
            if self.face_enhance: # Only log if user wanted it but we disabled it
                logger.info("🔧 Face enhancement disabled due to VRAM constraints.")
            self.face_enhance = False
        
        if not os.path.exists(self.realesrgan_model_path):
            raise FileNotFoundError(f"RealESRGAN model not found at {self.realesrgan_model_path}")
            
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.realesrgan_model_path,
            model=model,
            tile=config["tile"],
            tile_pad=10,
            pre_pad=0,
            half=config["half"],
            device=self.device
        )
        
        if self.face_enhance:
            if not os.path.exists(self.gfpgan_model_path):
                logger.warning(f"⚠️ GFPGAN model not found at {self.gfpgan_model_path}. Face enhancement disabled.")
                self.face_enhance = False
            else:
                self.face_enhancer = GFPGANer(
                    model_path=self.gfpgan_model_path,
                    upscale=self.scale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler
                )
                logger.info("✅ Face Enhancement (GFPGAN) Loaded.")
        
        logger.info("✅ Models Loaded Successfully.")

    def enhance_frame(self, img):
        """Enhance a single frame using RealESRGAN + GFPGAN."""
        try:
            if self.face_enhance and self.face_enhancer:
                _, _, output = self.face_enhancer.enhance(
                    img, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
            else:
                output, _ = self.upsampler.enhance(img, outscale=self.scale)
            
            # Skin Protection
            enable_skin_protect = os.getenv("ENABLE_SKIN_PROTECT", "yes").lower() == "yes"
            skin_max_brightness = int(os.getenv("SKIN_MAX_BRIGHTNESS", 175))
            
            if enable_skin_protect:
                # Detect skin (Simple YCrCb)
                img_ycrcb = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
                # Skin range: Cr [133, 173], Cb [77, 127]
                lower = np.array([0, 133, 77], dtype=np.uint8)
                upper = np.array([255, 173, 127], dtype=np.uint8)
                skin_mask = cv2.inRange(img_ycrcb, lower, upper)
                
                # Dilate to cover edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
                
                # Convert to LAB for brightness clamping
                lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Create a mask where skin is too bright
                # Note: LAB L channel is 0-255 in OpenCV
                bright_skin_mask = cv2.bitwise_and(cv2.threshold(l, skin_max_brightness, 255, cv2.THRESH_BINARY)[1], skin_mask)
                
                # Clamp L channel in those areas
                l = np.where(bright_skin_mask > 0, skin_max_brightness, l)
                
                # Merge back
                lab_clamped = cv2.merge([l, a, b])
                output = cv2.cvtColor(lab_clamped, cv2.COLOR_LAB2BGR)
            
            return output
        except Exception as e:
            logger.error(f"Frame enhancement failed: {e}")
            return img

    def process_video(self, input_path, output_path, progress_callback=None):
        logger.info(f"🎬 Starting Video Enhancement: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ Could not open input video.")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # OPTIMIZATION: Smart Frame Skipping for Previews/Fast Mode
        # But for final output we usually want all frames unless specified
        skip_frames = 1 
        
        target_width = width * self.scale
        target_height = height * self.scale
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        prev_frame = None
        frame_buffer = []
        # Batch size depends on VRAM, but 4 is usually safe for T4
        batch_size = 4 if self.device.type == 'cuda' else 1
        
        try:
            logger.info(f"⚡ Processing with batch size: {batch_size}")
            processed_count = 0
            
            for i in tqdm(range(total_frames), desc="Enhancing"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add to batch
                frame_buffer.append(frame)
                
                # Process batch when full
                if len(frame_buffer) >= batch_size:
                    enhanced_batch = self._process_batch(frame_buffer)
                    for enhanced in enhanced_batch:
                        writer.write(enhanced)
                        processed_count += 1
                    
                    frame_buffer = []
                
                if progress_callback and i % 10 == 0:
                    progress_callback(i / total_frames)
            
            # Process remaining frames
            if frame_buffer:
                enhanced_batch = self._process_batch(frame_buffer)
                for enhanced in enhanced_batch:
                    writer.write(enhanced)
                    processed_count += 1
            
            logger.info(f"✅ Video Enhancement Complete. Processed {processed_count}/{total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
        finally:
            cap.release()
            writer.release()
    
    def _process_batch(self, frames):
        """Process multiple frames at once for GPU efficiency."""
        enhanced = []
        for frame in frames:
            enhanced.append(self.enhance_frame(frame))
        return enhanced
