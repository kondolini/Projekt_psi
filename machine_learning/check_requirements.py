#!/usr/bin/env python3
"""
Check if all required packages are installed for the ML training pipeline.
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package can be imported"""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def check_requirements() -> Tuple[List[str], List[str]]:
    """
    Check all required packages for the ML pipeline
    
    Returns:
        Tuple of (available_packages, missing_packages)
    """
    
    # Required packages with their import names
    requirements = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),  # Often needed with PyTorch
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pickle', 'pickle'),  # Built-in
        ('json', 'json'),      # Built-in
        ('logging', 'logging'), # Built-in
        ('datetime', 'datetime'), # Built-in
        ('collections', 'collections'), # Built-in
        ('re', 're'),          # Built-in
        ('os', 'os'),          # Built-in
        ('sys', 'sys'),        # Built-in
    ]
    
    available = []
    missing = []
    
    print("🔍 Checking package requirements...")
    
    for package_name, import_name in requirements:
        if check_package(package_name, import_name):
            available.append(package_name)
            print(f"✅ {package_name}")
        else:
            missing.append(package_name)
            print(f"❌ {package_name}")
    
    return available, missing

def check_pytorch_features():
    """Check PyTorch specific features"""
    try:
        import torch
        print(f"\n🔥 PyTorch version: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA version: {torch.version.cuda}")
            print(f"🔥 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  Training will use CPU only")
    except ImportError:
        print("❌ PyTorch not available")

def main():
    """Main function"""
    print("🚀 Greyhound Racing ML - Requirements Check")
    print("=" * 50)
    
    available, missing = check_requirements()
    
    print(f"\n📊 Summary:")
    print(f"   - Available: {len(available)} packages")
    print(f"   - Missing: {len(missing)} packages")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n📦 To install missing packages:")
        if 'torch' in missing:
            print("   # For PyTorch (CPU version):")
            print("   pip install torch torchvision torchaudio")
            print("   # Or for CUDA (check PyTorch website for your CUDA version):")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        other_missing = [p for p in missing if p not in ['torch', 'torchvision', 'torchaudio']]
        if other_missing:
            print(f"   # For other packages:")
            print(f"   pip install {' '.join(other_missing)}")
        
        print("\n📋 Or install from requirements file:")
        print("   pip install -r requirements_ai.txt")
        
        return False
    else:
        print("\n✅ All required packages are available!")
        check_pytorch_features()
        print("\n🎉 Your environment is ready for training!")
        return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
