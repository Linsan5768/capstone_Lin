#!/bin/bash

# Cattle2D Launcher
# Double-click this file to launch Cattle2D

echo "🚀 Launching Cattle2D..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if files exist
if [ ! -f "$SCRIPT_DIR/Cattle2D.py" ]; then
    echo "❌ Error: Cattle2D.py not found"
    echo "Please ensure this script is in the same directory as the application files"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check and install dependencies
echo "🔍 Checking dependencies..."
if [ -f "$SCRIPT_DIR/check_dependencies.py" ]; then
    python3 "$SCRIPT_DIR/check_dependencies.py"
    if [ $? -ne 0 ]; then
        echo "❌ Dependency check failed. Please fix the issues and try again."
        read -p "Press Enter to exit..."
        exit 1
    fi
else
    echo "⚠️  Dependency checker not found. Skipping dependency check."
    echo "If you encounter import errors, please run: pip install -r requirements.txt"
fi

echo ""

echo "📱 Starting Cattle2D..."
echo "💡 Tips:"
echo "   - The application may take a few seconds to initialize"
echo "   - If prompted for camera permission, click 'Allow'"
echo "   - The application window should appear automatically"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Launch application
echo "🔄 Starting..."
python3 Cattle2D.py

echo ""
echo "✅ Application has exited"
read -p "Press Enter to close this window..."
