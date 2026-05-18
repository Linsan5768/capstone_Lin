#!/bin/bash

# Quick build script for Human Height Detection Apps
# This is a simplified version of build_apps.sh for quick builds

set -e

echo "🚀 Quick Build: Human Height Detection Apps"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Clean and build
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/

# Build Demo App
echo "🔨 Building HumanHeightDemo.app..."
python3 -m PyInstaller demo.spec --clean --noconfirm

# Build Depth App
echo "🔨 Building DepthHumanHeight.app..."
python3 -m PyInstaller depth_human_height.spec --clean --noconfirm

echo "✅ Build completed!"
echo "📱 Apps are in the dist/ directory"
echo "   • HumanHeightDemo.app"
echo "   • DepthHumanHeight.app"
