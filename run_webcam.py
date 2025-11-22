#!/usr/bin/env python3
"""
Simple webcam emotion detection runner
Just run: python run_webcam.py
"""

import subprocess
import sys

print("""
╔════════════════════════════════════════════════════════════════╗
║           🎭 EMOTION DETECTION - WEBCAM MODE                  ║
╚════════════════════════════════════════════════════════════════╝

Starting webcam...
Press 'q' in the display window to quit
Showing FPS in console
""")

try:
    subprocess.run([
        sys.executable, 'main.py',
        '--source', '0',
        '--show-fps'
    ])
except KeyboardInterrupt:
    print("\n\n👋 Stopped by user")
    sys.exit(0)
