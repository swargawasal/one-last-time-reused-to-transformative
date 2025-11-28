"""
Text Overlay Module
Handles adding text to videos with robust font detection and platform compatibility.
"""

import os
import subprocess
import logging
import platform
import shutil

logger = logging.getLogger("text_overlay")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

class TextOverlay:
    def __init__(self):
        pass # Font detection done per call or cached if needed

    def _get_font_path(self):
        """
        Resolve absolute font path.
        FIX: Copy font to temp dir to avoid Windows drive letter escaping issues (C:/ vs C\:/).
        """
        system_font = 'C:/Windows/Fonts/arial.ttf'
        if not os.path.exists(system_font):
            return "arial" # Fallback to internal name
            
        # Copy to temp folder to avoid drive letter issues in FFmpeg filter string
        temp_font = os.path.join("temp", "arial.ttf")
        try:
            if not os.path.exists(temp_font):
                shutil.copy(system_font, temp_font)
            return temp_font.replace("\\", "/")
        except Exception as e:
            logger.warning(f"Failed to copy font: {e}")
            return "arial"

    def _escape_text(self, text):
        """
        Escape text for FFmpeg drawtext.
        """
        # Escape single quotes and colons
        if not text:
            return ""
        text = text.replace("'", "'\''") # Escape single quote for shell/ffmpeg
        text = text.replace(":", "\\:")  # Escape colon for filter
        return text

    def add_overlay(self, video_path, output_path, text, position="bottom", size=60):
        """
        Apply text overlay using FFmpeg.
        """
        font_path = self._get_font_path()
        safe_text = self._escape_text(text)
        
        # Calculate position
        # bottom: x=(w-text_w)/2:y=h-h*0.15
        # top: x=(w-text_w)/2:y=h*0.15
        # center: x=(w-text_w)/2:y=(h-text_h)/2
        
        if position == "top":
            y_expr = "h*0.15"
        elif position == "center":
            y_expr = "(h-text_h)/2"
        elif position == "bottom_low":
            # Place lower than standard bottom (for stacking)
            # Standard bottom is h*0.85. This puts it at h*0.92
            y_expr = "h-h*0.08"
        else: # bottom
            y_expr = "h-h*0.15"
            
        # Construct filter string with safe quoting
        # fontfile='PATH'
        # Keep drawtext format EXACTLY as requested, but use local path
        
        # Check if box is enabled
        use_box = os.getenv("TEXT_OVERLAY_BOX", "yes").lower() == "yes"
        box_params = "box=1:boxcolor=black@0.6:boxborderw=10:" if use_box else ""

        drawtext = (
            f"drawtext=fontfile='{font_path}':text='{safe_text}':"
            f"fontsize={size}:fontcolor=white:borderw=2:bordercolor=black:"
            f"{box_params}"
            f"x=(w-text_w)/2:y={y_expr}"
        )
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vf", drawtext,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Text overlay failed: {e}")
            return False

# Global Instance
overlay_engine = TextOverlay()

def apply_text_overlay_safe(input_path: str, output_path: str, text: str, position: str = "bottom", size: int = 60) -> bool:
    return overlay_engine.add_overlay(input_path, output_path, text, position, size)
