#!/usr/bin/env python3
"""
Setup script to help users configure their environment for the ML training pipeline.
This script helps resolve common import and path issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def get_project_root() -> Path:
    """Find the project root directory"""
    current = Path(__file__).resolve()
    
    # Look for key project files to identify root
    for parent in current.parents:
        if (parent / "README.md").exists() and (parent / "models").exists():
            return parent
    
    # Fallback: assume parent of machine_learning folder
    return current.parent.parent

def setup_python_path():
    """Setup Python path for imports"""
    project_root = get_project_root()
    
    print(f"🎯 Project root: {project_root}")
    
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print("✅ Added project root to Python path")
    else:
        print("ℹ️  Project root already in Python path")
    
    return project_root

def check_data_directories(project_root: Path):
    """Check if data directories exist"""
    data_dir = project_root / "data"
    
    required_dirs = [
        data_dir / "dogs_enhanced",
        data_dir / "races", 
        data_dir / "unified"
    ]
    
    print(f"\n📁 Checking data directories in: {data_dir}")
    
    missing_dirs = []
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path.name}")
        else:
            print(f"❌ {dir_path.name}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing data directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\nℹ️  Make sure you have run the data preparation scripts first.")
        return False
    else:
        print("\n✅ All data directories found!")
        return True

def test_imports():
    """Test if our modules can be imported"""
    print(f"\n🧪 Testing imports...")
    
    try:
        from models.race import Race
        print("✅ models.race")
    except ImportError as e:
        print(f"❌ models.race: {e}")
        return False
    
    try:
        from models.dog import Dog  
        print("✅ models.dog")
    except ImportError as e:
        print(f"❌ models.dog: {e}")
        return False
    
    try:
        from machine_learning.model import GreyhoundRacingModel
        print("✅ machine_learning.model")
    except ImportError as e:
        print(f"❌ machine_learning.model: {e}")
        return False
    
    try:
        from machine_learning.dataset import GreyhoundDataset
        print("✅ machine_learning.dataset")
    except ImportError as e:
        print(f"❌ machine_learning.dataset: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

def check_pytorch():
    """Check PyTorch installation"""
    print(f"\n🔥 Checking PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available (version {torch.version.cuda})")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
        else:
            print("ℹ️  CUDA not available (CPU-only training)")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("📦 Install with: pip install torch torchvision torchaudio")
        return False

def run_quick_test(project_root: Path):
    """Run a quick test of the model"""
    print(f"\n🧪 Running quick model test...")
    
    try:
        test_script = project_root / "machine_learning" / "test_model.py"
        if test_script.exists():
            # Run the test script
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                print("✅ Model test passed!")
                # Show key output lines
                for line in result.stdout.split('\n'):
                    if 'MODEL DETAILS' in line or 'parameters' in line or 'Tests Passed' in line:
                        print(f"   {line}")
                return True
            else:
                print("❌ Model test failed!")
                print(f"Error: {result.stderr}")
                return False
        else:
            print("⚠️  test_model.py not found")
            return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Greyhound Racing ML - Environment Setup")
    print("=" * 50)
    
    # Setup Python path
    project_root = setup_python_path()
    
    # Check PyTorch
    pytorch_ok = check_pytorch()
    
    # Check data directories
    data_ok = check_data_directories(project_root)
    
    # Test imports
    imports_ok = test_imports()
    
    # Run quick test if everything looks good
    if pytorch_ok and imports_ok:
        test_ok = run_quick_test(project_root)
    else:
        test_ok = False
    
    print(f"\n📊 Setup Summary:")
    print(f"   - PyTorch: {'✅' if pytorch_ok else '❌'}")
    print(f"   - Data directories: {'✅' if data_ok else '❌'}")
    print(f"   - Imports: {'✅' if imports_ok else '❌'}")
    print(f"   - Model test: {'✅' if test_ok else '❌'}")
    
    if all([pytorch_ok, data_ok, imports_ok, test_ok]):
        print(f"\n🎉 Setup completed successfully!")
        print(f"📝 You can now run training with:")
        print(f"   cd {project_root}")
        print(f"   python machine_learning/train.py --epochs 10 --batch_size 16")
        return True
    else:
        print(f"\n❌ Setup incomplete. Please fix the issues above.")
        if not pytorch_ok:
            print("   • Install PyTorch: pip install torch torchvision torchaudio")
        if not data_ok:
            print("   • Run data preparation scripts to create required directories")
        if not imports_ok:
            print("   • Check that you're running from the correct directory")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
