
import os
import logging
import platform
import shutil
import json
import time

logger = logging.getLogger("health")

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

def check_cpu_thermal(safe_mode=False):
    """
    Check CPU usage and temperature (if available).
    Returns: (is_safe, reason)
    """
    if not psutil:
        return True, "psutil_missing"
    
    # Check CPU Usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    warn_percent = float(os.getenv("CPU_USAGE_WARN_PERCENT", "85"))
    
    if cpu_percent > warn_percent:
        return False, f"high_cpu_usage_{cpu_percent}%"
        
    # Check Temperature (Linux only usually)
    if hasattr(psutil, "sensors_temperatures"):
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > float(os.getenv("CPU_TEMP_WARN_C", "85")):
                        return False, f"high_cpu_temp_{entry.current}C"

    return True, "ok"

def check_gpu_vram():
    """
    Check GPU VRAM availability.
    Returns: (total_mb, free_mb)
    """
    if not torch or not torch.cuda.is_available():
        return 0, 0
        
    try:
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024 * 1024)
        
        # approximate free memory
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        free = total - allocated # simplistic
        
        return total, free
    except Exception as e:
        logger.error(f"VRAM check failed: {e}")
        return 0, 0

def check_health():
    """
    Run a full system health check.
    Returns: dict
    """
    status = {
        "timestamp": time.time(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cpu_safe": True,
        "gpu_available": False,
        "vram_mb": 0,
        "ram_free_mb": 0
    }
    
    # CPU
    is_safe, reason = check_cpu_thermal()
    status["cpu_safe"] = is_safe
    status["cpu_reason"] = reason
    
    # RAM
    if psutil:
        mem = psutil.virtual_memory()
        status["ram_free_mb"] = mem.available / (1024 * 1024)
        
    # GPU
    if torch and torch.cuda.is_available():
        status["gpu_available"] = True
        status["gpu_name"] = torch.cuda.get_device_name(0)
        total, free = check_gpu_vram()
        status["vram_total_mb"] = total
        status["vram_free_mb"] = free
        
    return status

def print_health_summary():
    h = check_health()
    logger.info(f"🏥 Health Check: CPU Safe={h['cpu_safe']} ({h.get('cpu_reason')}), GPU={h['gpu_available']}, RAM Free={h['ram_free_mb']:.0f}MB")
    return h
