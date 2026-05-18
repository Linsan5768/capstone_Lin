#!/bin/bash

# DepthCattleHeight Launcher
# Double-click this file to launch DepthCattleHeight

echo "🚀 Launching DepthCattleHeight..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if files exist
if [ ! -f "$SCRIPT_DIR/DepthCattleHeight.py" ]; then
    echo "❌ Error: DepthCattleHeight.py not found"
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

echo "📱 Starting DepthCattleHeight..."
echo "💡 Tips:"
echo "   - The application may take a few seconds to initialize"
echo "   - Ensure OAK-D depth camera is connected via USB"
echo "   - Check USB cable and connection"
echo "   - The application window should appear automatically"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Launch application
echo "🔄 Starting with optimized parameters..."
python3 DepthCattleHeight.py

echo ""
echo "✅ Application has exited"
echo "💡 If you encountered issues, check:"
echo "- OAK-D camera connection"
echo "- USB cable"
echo "- Camera permissions"
read -p "Press Enter to close this window..."
