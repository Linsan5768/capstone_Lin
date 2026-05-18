# 📦 Installation Guide

## Quick Installation

1. **Download** the application files to your Mac
2. **First Time Setup**: Double-click `InstallDependencies.command` to install all required dependencies
3. **Launch Application**: Double-click one of the launcher files:
   - `LaunchCattle2D.command` - For camera-based detection
   - `LaunchDepthCattleHeight.command` - For OAK-D depth camera detection
   - `LaunchApplication.command` - Universal launcher

## Manual Installation

If the automatic installer doesn't work, you can install dependencies manually:

```bash
# Navigate to the project directory
cd /path/to/your/project

# Install dependencies
pip3 install -r requirements.txt
```

## System Requirements

- **macOS**: 10.15 or higher
- **Hardware**: 
  - Camera (for Cattle2D)
  - OAK-D depth camera (for DepthCattleHeight)

## First Launch

1. Double-click the launcher file
2. If prompted about security, right-click the file and select "Open"
3. Allow camera permissions when requested
4. Wait for the application to load (may take a few seconds)

## Troubleshooting

- **"Cannot verify developer"**: Right-click → "Open"
- **No window appears**: Wait a few seconds, check Dock for app icon
- **Camera issues**: Check System Preferences → Security & Privacy → Camera

## Support

For technical support, please provide:
- macOS version
- Error messages
- Hardware configuration
