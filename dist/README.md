# 🎉 Cattle Height Detection Application

A comprehensive cattle height detection system with two different approaches: camera-based real-time detection and OAK-D depth camera-based 3D detection.

## 🚀 Quick Start

### First Time Setup (Required)
1. **Double-click `InstallDependencies.command`** - Install all required dependencies
2. Wait for installation to complete (may take a few minutes)

### Method 1: Use Launcher Scripts (Recommended)
1. **Double-click `LaunchCattle2D.command`** - Launch camera-based real-time height detection
2. **Double-click `LaunchDepthCattleHeight.command`** - Launch OAK-D depth camera-based 3D height detection
3. **Double-click `LaunchApplication.command`** - Choose which application to launch

### Method 2: Direct Python Launch
- **Run `python3 Cattle2D.py`** - Launch Cattle2D directly
- **Run `python3 DepthCattleHeight.py`** - Launch DepthCattleHeight directly

### Method 3: Manual Dependency Installation
If the automatic installer doesn't work, you can install dependencies manually:
```bash
pip3 install -r requirements.txt
```

## 📱 Application Description

### Cattle2D
- **Function**: Real-time cattle height detection using camera
- **Technology**: YOLO11n-seg + AprilTag + Manual ground calibration
- **Use Case**: Regular camera environment, requires AprilTag calibration

### DepthCattleHeight
- **Function**: 3D cattle height detection using OAK-D depth camera
- **Technology**: DepthAI + YOLO + 3D point cloud processing
- **Use Case**: Professional depth camera environment, more precise 3D measurement


## 🔧 System Requirements

- **Operating System**: macOS 10.15 or higher
- **Python**: 3.9+ (included in the application)
- **Hardware**: 
  - Cattle2D: Regular camera
  - DepthCattleHeight: OAK-D depth camera

## 🎯 Usage Instructions

### Launching the Application
1. Double-click the corresponding launcher file (`.command` file)
2. Wait for the application to load (may take a few seconds)
3. The application window will appear automatically

### Basic Operations
- **Exit**: Press `q` key or `ESC` key
- **Fullscreen**: Press `f` key
- **Resize window**: Press `+`/`-` keys

## 🚨 Troubleshooting

### Problem 1: Double-clicking launcher has no response
**Solution**:
1. Ensure the file has execution permissions
2. Right-click the file → "Open"
3. Allow execution in security settings

### Problem 2: Application launches but no window appears
**Solution**:
1. Wait a few seconds for the application to fully load
2. Check if there's an application icon in the Dock
3. Press `Cmd+Tab` to switch applications
4. Check camera permission settings

### Problem 3: "Cannot verify developer" prompt
**Solution**:
1. Right-click the file → "Open"
2. Allow execution in security settings

### Problem 4: Camera permission issues
**Solution**:
1. System Preferences → Security & Privacy → Camera
2. Ensure the application has camera access permission


## 🧪 Testing Tools

### Application Test
```bash
python3 HumanHeightDemo.py --help      # Test HumanHeightDemo
python3 DepthHumanHeight.py --help     # Test DepthHumanHeight
```

## 📞 Technical Support

If you encounter problems, please:
1. Check console error messages
2. Check system permission settings
3. Ensure hardware connections are normal
4. Contact technical support with detailed error information

## ✅ Verify Application Works Properly

Test the applications:
```bash
python3 HumanHeightDemo.py --help
python3 DepthHumanHeight.py --help
```

If the help messages display correctly, the applications are working properly.

## 📁 File Structure

```
dist/
├── LaunchCattle2D.command             # 🎯 Recommended launch method
├── LaunchDepthCattleHeight.command    # 🎯 Recommended launch method
├── LaunchApplication.command          # Universal launcher
├── Cattle2D.py                        # Python launcher
├── DepthCattleHeight.py               # Python launcher
├── INSTALL.md                         # Installation guide
└── README.md                          # This file
```

## 🎯 User Recommendations

**Simplest Method**:
1. Double-click `LaunchCattle2D.command` or `LaunchDepthCattleHeight.command`
2. Or use `LaunchApplication.command` to choose which application to launch
3. These launchers display status information to help users understand the application startup process
4. If problems occur, the launcher will display error messages

**If launchers don't work**:
1. Right-click the `.command` file → "Open"
2. Allow execution in security settings
3. Use direct Python launch: `python3 Cattle2D.py` or `python3 DepthCattleHeight.py`

---

**Remember**: We recommend using `.command` launcher files as they provide better error handling and user feedback!