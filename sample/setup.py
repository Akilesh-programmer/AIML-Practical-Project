#!/usr/bin/env python3
"""
Setup script for Universal Multi-Metal Alloy Optimizer
Installs all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Universal Multi-Metal Alloy Optimizer - Setup")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ ERROR: Python 3.8+ is required")
        print("Please upgrade your Python installation")
        return False
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "requirements.txt").exists():
        print("❌ ERROR: requirements.txt not found")
        print("Make sure you're running this from the sample/ directory")
        return False
    
    print(f"📁 Installation directory: {current_dir}")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: Could not upgrade pip, continuing anyway...")
    
    # Install minimal requirements first
    if (current_dir / "requirements-minimal.txt").exists():
        print("\n📦 Installing minimal requirements...")
        if not run_command(f"{sys.executable} -m pip install -r requirements-minimal.txt", "Installing minimal requirements"):
            print("❌ Failed to install minimal requirements")
            return False
    
    # Install full requirements
    print("\n📦 Installing full requirements...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing full requirements"):
        print("❌ Failed to install full requirements")
        print("💡 Try installing minimal requirements only:")
        print(f"   {sys.executable} -m pip install -r requirements-minimal.txt")
        return False
    
    # Verify critical imports
    print("\n🔍 Verifying installation...")
    critical_modules = [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("joblib", "joblib"),
        ("requests", "requests")
    ]
    
    all_good = True
    for module, package in critical_modules:
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - FAILED")
            all_good = False
    
    if all_good:
        print("\n🎉 Installation completed successfully!")
        print("\n🚀 Ready to start the Universal Alloy Optimizer:")
        print("   python quick_alloy_api.py")
        print("   OR")
        print("   python start_server.py")
        print("\n🌐 Server will be available at: http://localhost:8001")
        return True
    else:
        print("\n❌ Some packages failed to install")
        print("💡 Try manual installation:")
        print("   pip install scikit-learn fastapi uvicorn pydantic")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)