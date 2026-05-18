#!/bin/bash

# Cattle Detection System Installation Script

echo "=== Cattle Detection System Installation Script ==="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Found Python: $python_version"
else
    echo "✗ Python3 not found, please install Python 3.8+"
    exit 1
fi

# Check pip
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ Found pip3"
else
    echo "✗ pip3 not found, please install pip"
    exit 1
fi

# Create virtual environment (optional)
read -p "Create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv cattle_detection_env
    source cattle_detection_env/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Download YOLO model
echo "Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

if [[ $? -eq 0 ]]; then
    echo "✓ YOLO model downloaded successfully"
else
    echo "⚠ YOLO model download failed, will attempt auto-download on first run"
fi

# Set execution permissions
echo "Setting script execution permissions..."
chmod +x run_detection.py
chmod +x test_system.py

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Usage:"
echo "1. Basic Usage:"
echo "   python3 run_detection.py"
echo ""
echo "2. Use Enhanced Mode:"
echo "   python3 run_detection.py --enhanced"
echo ""
echo "3. Run Tests:"
echo "   python3 run_detection.py --test"
echo ""
echo "4. View Help:"
echo "   python3 run_detection.py --help"
echo ""

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Note: If you created a virtual environment, remember to activate it before each use:"
    echo "   source cattle_detection_env/bin/activate"
fi
