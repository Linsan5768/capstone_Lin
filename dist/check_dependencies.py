#!/usr/bin/env python3
"""
Dependency checker and installer for Cattle Height Detection Application
"""

import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path

# Required packages from requirements.txt
REQUIRED_PACKAGES = [
    'opencv-python>=4.8.0',
    'numpy>=1.24.0',
    'ultralytics>=8.0.0',
    'torch>=2.0.0',
    'Pillow>=10.0.0',
    'matplotlib>=3.7.0',
    'PyYAML>=6.0',
    'requests>=2.31.0',
    'tqdm>=4.65.0',
    'uvicorn[standard]>=0.35.0',
    'fastapi>=0.116.1',
    'PyQt6>=6.9.1',
    'PySide6>=6.7.0',
    'depthai==2.30.0.0'
]

def check_package_installed(package_spec):
    """Check if a package is installed and meets version requirements"""
    try:
        # Parse package name and version
        if '>=' in package_spec:
            name, version = package_spec.split('>=')
            name = name.strip()
            version = version.strip()
        elif '==' in package_spec:
            name, version = package_spec.split('==')
            name = name.strip()
            version = version.strip()
        else:
            name = package_spec.strip()
            version = None
        
        # Check if package is installed
        try:
            installed_version = pkg_resources.get_distribution(name).version
            if version:
                # Check version requirement
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                    return False, f"Version {installed_version} < required {version}"
            return True, installed_version
        except pkg_resources.DistributionNotFound:
            return False, "Not installed"
    except Exception as e:
        return False, f"Error checking: {e}"

def install_package(package_spec):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package_spec}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_spec
        ], capture_output=True, text=True, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Installation failed: {e.stderr}"

def check_and_install_dependencies():
    """Check and install all required dependencies"""
    print("🔍 Checking dependencies...")
    print("=" * 50)
    
    missing_packages = []
    outdated_packages = []
    
    # Check each required package
    for package in REQUIRED_PACKAGES:
        is_installed, info = check_package_installed(package)
        if is_installed:
            print(f"✅ {package} - {info}")
        else:
            print(f"❌ {package} - {info}")
            missing_packages.append(package)
    
    print("=" * 50)
    
    # Install missing packages
    if missing_packages:
        print(f"📦 Found {len(missing_packages)} missing packages. Installing...")
        print("")
        
        for package in missing_packages:
            success, message = install_package(package)
            if success:
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}: {message}")
                return False
        
        print("")
        print("✅ All dependencies installed successfully!")
    else:
        print("✅ All dependencies are already installed!")
    
    return True

def main():
    """Main function"""
    print("🚀 Cattle Height Detection - Dependency Checker")
    print("=" * 50)
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent
    requirements_file = script_dir.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found in parent directory")
        print(f"Expected: {requirements_file}")
        input("Press Enter to exit...")
        return False
    
    # Check and install dependencies
    success = check_and_install_dependencies()
    
    if success:
        print("")
        print("🎉 Dependency check completed successfully!")
        print("You can now run the application.")
    else:
        print("")
        print("❌ Dependency installation failed.")
        print("Please check the error messages above and try again.")
    
    print("")
    input("Press Enter to continue...")
    return success

if __name__ == "__main__":
    main()
