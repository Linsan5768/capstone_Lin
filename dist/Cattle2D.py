#!/usr/bin/env python3
"""
Cattle2D starter
Provide an interactive interface: select the hip height measurement method and the AprilTag distance, and then start the actual detection program
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    src_dir = project_root / "src"

    # --- Ensure correct Python path ---
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(src_dir), str(project_root), os.environ.get("PYTHONPATH", "")]
    )
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    os.environ["MPLBACKEND"] = "TkAgg"

    print("🐄 Welcome to Cattle2D Real-time Height Detection")
    print("--------------------------------------------------")
    print("Please select the measurement formula method:")
    print("1) Bounding Box")
    print("2) Ground (*Recommended)")
    print("")

    formula_choice = input("Enter your choice (1-2): ").strip()
    if formula_choice not in ["1", "2"]:
        print("❌ Invalid selection. Exiting.")
        input("Press Enter to exit...")
        return

    tag_distance = input(
        "Enter the real-world distance between two AprilTags (in meters, e.g., 1.13): "
    ).strip()
    if not tag_distance:
        tag_distance = "1.13"  # default
    print("")

    # --- Build the command ---
    height_method = "bbox" if formula_choice == "1" else "ground"

    cmd = [
        "python3",
        "src/cli/demo.py",
        "--oak",
        "--model",
        "yolo11n-seg.pt",
        "--confidence",
        "0.1",
        "--tag-distance",
        tag_distance,
        "--height-method",
        height_method,
    ]

    env = os.environ.copy()
    env["DEPTHAI_FORCE_USB2"] = "1"

    print("🚀 Launching detection pipeline...")
    print(f"   Formula method: {height_method}")
    print(f"   Tag distance: {tag_distance} m")
    print("--------------------------------------------------\n")

    try:
        subprocess.run(cmd, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Runtime error: {e}")
    finally:
        print("\n✅ Cattle2D has exited.")
        input("Press Enter to return to the launcher...")


if __name__ == "__main__":
    main()
