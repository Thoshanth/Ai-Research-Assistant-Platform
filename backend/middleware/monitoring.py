import time
import psutil
import os
from backend.logger import get_logger

logger = get_logger("middleware.monitoring")


def log_request(method: str, path: str, status_code: int, duration_ms: float):
    """
    Logs every HTTP request with timing information.
    This is how you spot slow endpoints in production.
    """
    level = "INFO" if status_code < 400 else "ERROR"
    msg = (
        f"HTTP | {method} {path} | "
        f"status={status_code} | "
        f"duration={duration_ms:.2f}ms"
    )
    if level == "INFO":
        logger.info(msg)
    else:
        logger.error(msg)


def get_system_stats() -> dict:
    """
    Returns current system resource usage.
    Used in the health check endpoint so you can see
    if the server is running out of memory or CPU.
    """
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "memory_percent": round(process.memory_percent(), 2),
        "disk_free_gb": round(
            psutil.disk_usage("/").free / (1024 ** 3), 2
        ),
    }