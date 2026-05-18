#!/bin/bash

# Dependency Installer for Cattle Height Detection Application
# Double-click this file to install all required dependencies

echo "🔧 Cattle Height Detection - Dependency Installer"
echo "=================================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if requirements.txt exists
REQUIREMENTS_FILE="$SCRIPT_DIR/../requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "Expected location: $REQUIREMENTS_FILE"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "📦 Installing dependencies from requirements.txt..."
echo "Location: $REQUIREMENTS_FILE"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 first: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Install dependencies
echo "🚀 Installing dependencies..."
echo "This may take a few minutes..."
echo ""

python3 -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "❌ Failed to upgrade pip"
    read -p "Press Enter to exit..."
    exit 1
fi

python3 -m pip install -r "$REQUIREMENTS_FILE"
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "🎉 You can now run the application using:"
    echo "   - LaunchApplication.command"
    echo "   - LaunchCattle2D.command"
    echo "   - LaunchDepthCattleHeight.command"
else
    echo ""
    echo "❌ Some dependencies failed to install"
    echo "Please check the error messages above"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "1. Make sure you have internet connection"
    echo "2. Try running: python3 -m pip install --upgrade pip"
    echo "3. Try running: python3 -m pip install -r requirements.txt --user"
    echo "4. Check if you have sufficient disk space"
fi

echo ""
read -p "Press Enter to exit..."
