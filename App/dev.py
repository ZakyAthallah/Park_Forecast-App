"""
Dev watcher — auto-restarts parking_forecast_legacy.py on save.
Usage:  python dev.py
"""
import subprocess
import sys
import time
from pathlib import Path

TARGET = Path(__file__).parent / "parking_forecast_legacy.py"

def start():
    return subprocess.Popen([sys.executable, str(TARGET)])

mtime = TARGET.stat().st_mtime
proc  = start()
print(f"[dev] Watching {TARGET.name} — Ctrl+C to stop")

try:
    while True:
        time.sleep(0.8)
        new_mtime = TARGET.stat().st_mtime
        if new_mtime != mtime:
            mtime = new_mtime
            print("[dev] Change detected — restarting...")
            proc.kill()
            proc.wait()
            proc = start()
except KeyboardInterrupt:
    print("[dev] Stopping.")
    proc.kill()
