#!/usr/bin/env python3
"""
DepthCattleHeight Launcher
Provides preconfigured OAK-D depth camera parameters and environment setup.
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    src_dir = project_root / "src"

    # --- Ensure correct import path ---
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(src_dir), str(project_root), os.environ.get("PYTHONPATH", "")]
    )
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    os.environ["MPLBACKEND"] = "TkAgg"
    os.environ["DEPTHAI_FORCE_USB2"] = "1"  # <-- Important for MacBook USB2 compatibility

    print("🌊 DepthCattleHeight Launcher — OAK-D 3D Height Measurement")
    print("------------------------------------------------------------")
    print("Launching with preconfigured parameters...")
    print("")

    # --- Build argument list ---
    cmd = [
        "python3",
        "src/cli/depth_human_hight.py",
        "--source", "oak",
        "--fps", "25",
        "--oak_rgb_res", "1080p",
        "--oak_stereo_res", "400p",
        "--oak_lr_check",
        "--oak_subpixel",
        "--conf_thr", "180",
        "--smart_model", "yolo11n-seg.pt",
        "--smart_conf", "0.30",
        "--smart_target", "both",
        "--groundline_stride", "6",
        "--height_k", "0.60",
    ]

    print("💡 Parameters:")
    print("   - FPS: 25")
    print("   - RGB Resolution: 1080p")
    print("   - Stereo Resolution: 400p")
    print("   - Left-Right Check: Enabled")
    print("   - Subpixel: Enabled")
    print("   - Confidence Threshold: 180")
    print("   - YOLO Model: yolo11n-seg.pt")
    print("   - Smart Confidence: 0.30")
    print("   - Smart Target: both")
    print("   - Groundline Stride: 6")
    print("   - Height Scaling K: 0.60")
    print("------------------------------------------------------------\n")

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Subprocess exited with error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        print("")
        print("🔧 Troubleshooting tips:")
        print("1. Ensure OAK-D camera is connected via USB")
        print("2. Check USB cable and connection")
        print("3. Check camera permissions in System Preferences")
        print("4. Restart the OAK-D camera")
    finally:
        print("\n✅ DepthCattleHeight has exited.")
        input("Press Enter to return to the launcher...")


if __name__ == "__main__":
    main()
