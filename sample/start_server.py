#!/usr/bin/env python3
"""
Quick Start Script for Universal Multi-Metal Alloy Optimizer
Run this to start the working API server
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting Universal Multi-Metal Alloy Optimizer")
    print("="*60)
    
    # Check if we're in the sample directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if models exist
    models_dir = current_dir / "models"
    if not models_dir.exists():
        print("❌ ERROR: models/ directory not found!")
        print("Make sure you're running this from the sample/ directory")
        return
    
    # Count model files
    model_files = list(models_dir.glob("*.pkl"))
    print(f"📊 Found {len(model_files)} trained models:")
    for model in model_files:
        print(f"   ✅ {model.name}")
    
    print(f"\n🔄 Starting API server...")
    print(f"🌐 Server will run on: http://localhost:8001")
    print(f"📡 Endpoint: POST http://localhost:8001/optimize")
    print(f"🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Start the API server
        subprocess.run([sys.executable, "quick_alloy_api.py"], check=True)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
    except FileNotFoundError:
        print(f"\n❌ Error: quick_alloy_api.py not found in current directory")

if __name__ == "__main__":
    main()