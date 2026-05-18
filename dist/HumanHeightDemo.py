#!/usr/bin/env python3
"""
Cattle2D starter
Solve the startup problem of GUI applications packaged by PyInstaller on macOS
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Set the Python path
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['QT_QPA_PLATFORM'] = 'cocoa'
    os.environ['MPLBACKEND'] = 'TkAgg'
    
    # Import and run the original script
    try:
        # Switch to the project root directory
        os.chdir(project_root)
        
        # Import the original module
        from cli.demo import main as demo_main
        
        # Run the main function
        demo_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed")
        input("Press enter to exit...")
    except Exception as e:
        print(f"❌ Running error: {e}")
        input("Press enter to exit...")

if __name__ == "__main__":
    main()
