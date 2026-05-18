#!/bin/bash

# Human Height Detection Apps Builder for macOS
# This script builds both demo.py and depth_human_hight.py into .app bundles

set -e  # Exit on any error

echo "🚀 Building Human Height Detection Apps for macOS..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only!"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH!"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed or not in PATH!"
    exit 1
fi

print_status "Installing/updating dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    print_error "PyInstaller installation failed!"
    exit 1
fi

print_success "Dependencies installed successfully!"

# Create build directory
BUILD_DIR="build"
DIST_DIR="dist"
print_status "Creating build directories..."
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf "$BUILD_DIR"/* "$DIST_DIR"/*

# Build Demo App
print_status "Building HumanHeightDemo.app..."
if python3 -m PyInstaller demo.spec --clean --noconfirm; then
    print_success "HumanHeightDemo.app built successfully!"
    
    # Move to dist directory
    if [ -d "dist/HumanHeightDemo.app" ]; then
        mv "dist/HumanHeightDemo.app" "$DIST_DIR/"
        print_success "HumanHeightDemo.app moved to $DIST_DIR/"
    fi
else
    print_error "Failed to build HumanHeightDemo.app!"
    exit 1
fi

# Build Depth Human Height App
print_status "Building DepthHumanHeight.app..."
if python3 -m PyInstaller depth_human_height.spec --clean --noconfirm; then
    print_success "DepthHumanHeight.app built successfully!"
    
    # Move to dist directory
    if [ -d "dist/DepthHumanHeight.app" ]; then
        mv "dist/DepthHumanHeight.app" "$DIST_DIR/"
        print_success "DepthHumanHeight.app moved to $DIST_DIR/"
    fi
else
    print_error "Failed to build DepthHumanHeight.app!"
    exit 1
fi

# Create launcher scripts for easier execution
print_status "Creating launcher scripts..."

# Demo app launcher
cat > "$DIST_DIR/run_demo.sh" << 'EOF'
#!/bin/bash
# Launcher script for HumanHeightDemo.app
# This script sets up the environment and runs the demo

# Set environment variables
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Run the demo
exec "$(dirname "$0")/HumanHeightDemo.app/Contents/MacOS/HumanHeightDemo" "$@"
EOF

chmod +x "$DIST_DIR/run_demo.sh"

# Depth app launcher
cat > "$DIST_DIR/run_depth.sh" << 'EOF'
#!/bin/bash
# Launcher script for DepthHumanHeight.app
# This script sets up the environment and runs the depth human height detection

# Set environment variables
export DEPTHAI_FORCE_USB2=1
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Default arguments (can be overridden by command line)
DEFAULT_ARGS=(
    "--fps" "25"
    "--oak_rgb_res" "1080p"
    "--oak_stereo_res" "400p"
    "--oak_lr_check"
    "--oak_subpixel"
    "--conf_thr" "180"
    "--weights" "path/to/yolo11n.pt"
    "--groundline_stride" "6"
    "--height_k" "0.60"
)

# If no arguments provided, use defaults
if [ $# -eq 0 ]; then
    exec "$(dirname "$0")/DepthHumanHeight.app/Contents/MacOS/DepthHumanHeight" "${DEFAULT_ARGS[@]}"
else
    exec "$(dirname "$0")/DepthHumanHeight.app/Contents/MacOS/DepthHumanHeight" "$@"
fi
EOF

chmod +x "$DIST_DIR/run_depth.sh"

# Create a README for the built apps
cat > "$DIST_DIR/README.md" << 'EOF'
# Human Height Detection Apps

This directory contains the built macOS applications for human height detection.

## Applications

### 1. HumanHeightDemo.app
- **Purpose**: Human height detection using YOLO11n-seg + AprilTag + manual ground + CSV logging
- **Usage**: Double-click the app or use the launcher script
- **Launcher**: `./run_demo.sh`

### 2. DepthHumanHeight.app
- **Purpose**: OAK RGB+Depth visual debugger with YOLO head-top + 3D height
- **Usage**: Double-click the app or use the launcher script
- **Launcher**: `./run_depth.sh`
- **Default Settings**: 
  - FPS: 25
  - RGB Resolution: 1080p
  - Stereo Resolution: 400p
  - Left-Right Check: Enabled
  - Subpixel: Enabled
  - Confidence Threshold: 180
  - Ground Line Stride: 6
  - Height K: 0.60

## Requirements

- macOS 10.15 or later
- OAK-D camera (for DepthHumanHeight.app)
- YOLO model files (included in the apps)

## Usage

### Running the Apps

1. **Double-click method**: Simply double-click on the .app files
2. **Command line method**: Use the provided launcher scripts
3. **Direct execution**: Navigate to the app contents and run the executable

### Command Line Options

Both apps support various command line options. Run with `--help` to see all available options:

```bash
./run_demo.sh --help
./run_depth.sh --help
```

### Customizing DepthHumanHeight.app

You can override the default settings by passing arguments to the launcher:

```bash
./run_depth.sh --fps 30 --oak_rgb_res 720p --height_k 0.65
```

## Troubleshooting

1. **Permission Issues**: Make sure the apps have the necessary permissions
2. **Camera Issues**: Ensure the OAK-D camera is properly connected
3. **Model Files**: The YOLO model files are included in the apps
4. **Dependencies**: All required dependencies are bundled with the apps

## Support

For issues or questions, please refer to the main project documentation.
EOF

print_success "Build completed successfully!"
print_status "Built applications are located in: $DIST_DIR/"
print_status ""
print_status "📱 Applications built:"
print_status "  • HumanHeightDemo.app - YOLO + AprilTag height detection"
print_status "  • DepthHumanHeight.app - OAK depth-based height detection"
print_status ""
print_status "🚀 Launcher scripts created:"
print_status "  • run_demo.sh - Launcher for demo app"
print_status "  • run_depth.sh - Launcher for depth app (with default OAK settings)"
print_status ""
print_status "📖 Documentation: $DIST_DIR/README.md"
print_status ""
print_success "Ready to use! You can now distribute these .app files."
