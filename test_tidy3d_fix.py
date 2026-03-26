#!/usr/bin/env python
"""Test Tidy3D cloud simulation with encoding fix."""
import os
import sys

# Set UTF-8 environment BEFORE any imports
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Try to set console code page to UTF-8 (Windows)
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

print(f"TIDY3D_API_KEY set: {bool(os.getenv('TIDY3D_API_KEY'))}")

# Run the simulation
from PhotonicsAI.Photon.tidy3d_runner import run_tidy3d_simulation
run_tidy3d_simulation('mmi', 1550.0, 0.5, 0.22, 10.0, 5.0, 0.2, 2.5, 10.0)