#!/bin/bash

# Cattle Height Detection Application Launcher
# Double-click this file to launch the application

echo "🚀 Launching Cattle Height Detection Application..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check application files
if [ ! -f "$SCRIPT_DIR/Cattle2D.py" ] || [ ! -f "$SCRIPT_DIR/DepthCattleHeight.py" ]; then
    echo "❌ Error: Application files not found"
    echo "Please ensure this script is in the same directory as the application files"
    read -p "Press Enter to exit..."
    exit 1
fi

while true; do
    echo "Please select the application to launch:"
    echo "1) Cattle2D (Real-time height detection using camera, Apriltags ver.)"
    echo "2) DepthCattleHeight (3D height detection using OAK-D depth camera, depth ver.)"
    echo "3) Exit"
    echo ""

    read -p "Enter your choice (1-3): " choice

    case "$choice" in
        1)
            echo "Launching Cattle2D..."
            cd "$SCRIPT_DIR"
            python3 Cattle2D.py
            echo ""
            echo "✅ Cattle2D has exited."
            ;;
        2)
            echo "Launching DepthCattleHeight..."
            cd "$SCRIPT_DIR"
            python3 DepthCattleHeight.py
            echo ""
            echo "✅ DepthCattleHeight has exited."
            ;;
        3)
            echo "Exiting to terminal. Bye!"
            break
            ;;
        *)
            echo "Invalid choice. Please enter 1, 2, or 3."
            ;;
    esac

    echo ""
    echo "If the application didn't show a window, please check:"
    echo "- Camera permission settings"
    echo "- Hardware connections"
    echo "- Console error messages"
    echo ""
    read -p "Press Enter to return to the menu..."
    echo ""
done

# If this script was launched by Finder (no parent shell), keep window open for interaction
if [[ -z "$PS1" ]]; then
    echo ""
    echo "Returning to interactive shell..."
    exec "$SHELL" -l
fi
