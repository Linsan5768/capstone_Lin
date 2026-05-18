# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Get the project root directory
ROOT = Path.cwd()

block_cipher = None

a = Analysis(
    ['src/cli/demo.py'],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Contains YOLO model files
        ('src/cli/yolo11n-seg.pt', 'src/cli/'),
        # Contains the SORT module
        ('src/external/sort.py', 'src/external/'),
        # Contains requirements.txt is used for dependency checking
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.models.yolo.segment',
        'ultralytics.utils',
        'ultralytics.utils.torch_utils',
        'cv2',
        'numpy',
        'depthai',
        'depthai.node',
        'depthai.pipeline',
        'depthai.device',
        'PIL',
        'PIL.Image',
        'torch',
        'torchvision',
        'matplotlib',
        'matplotlib.pyplot',
        'yaml',
        'requests',
        'tqdm',
        'collections',
        'pathlib',
        'argparse',
        'time',
        'os',
        'csv',
        'deque',
        'sys',
        'filterpy',
        'filterpy.kalman',
        'filterpy.common',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_opencv.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HumanHeightDemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

app = BUNDLE(
    exe,
    name='HumanHeightDemo.app',
    icon=None,
    bundle_identifier='com.soft3888.humanheight.demo',
    info_plist={
        'CFBundleName': 'Human Height Demo',
        'CFBundleDisplayName': 'Human Height Demo',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleExecutable': 'HumanHeightDemo',
        'CFBundleIdentifier': 'com.soft3888.humanheight.demo',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSMinimumSystemVersion': '10.15',
        'LSBackgroundOnly': False,
    },
)
