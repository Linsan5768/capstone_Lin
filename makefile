# ---- config ----
PY?=python3
CAM?=0
CONF?=0.30
FPS?=30
SAVE_CONF?=0.85
SAVE_DIR?=outputs/high_conf
OUTPUT?=outputs/output.mp4
METHOD?=
CUDA?=0
DISPLAY?=1

# cross-platform venv + deps (no need to 'source' the venv)
VENV := venv

# paths differ on Windows
PYBIN := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
  PYBIN := $(VENV)/Scripts/python.exe
endif

# ---- flags ----

ifeq ($(CUDA),1)
  CUDA_FLAG=--use_cuda
endif
ifeq ($(DISPLAY),1)
  DISPLAY_FLAG=--display
endif
ifneq ($(METHOD),)
  METHOD_FLAG=--method $(METHOD)
endif

# ---- tasks ----
help:
	@echo "make setup               # create venv and install requirements"
	@echo "make video VIDEO=path.mp4 [CONF=0.3 FPS=30 DISPLAY=1 SAVE_CONF=0.85 SAVE_DIR=... OUTPUT=... CUDA=0]"
	@echo "make cam   [CAM=0 CONF=0.3 FPS=30 DISPLAY=1 CUDA=0]"
	@echo "make image IMG=path.jpg  [CONF=0.3 METHOD=seg]"
	@echo "make clean               # remove caches/artifacts"
	@echo ""
	@echo "macOS App Building:"
	@echo "make build-apps          # build both .app bundles with full setup"
	@echo "make build-demo          # build only HumanHeightDemo.app"
	@echo "make build-depth         # build only DepthHumanHeight.app"
	@echo "make quick-build         # quick build of both apps"
	@echo "make install-deps        # install dependencies for app building"
	@echo "make clean-apps          # clean app build artifacts"

setup:
	$(PY) -m venv $(VENV)
	$(PYBIN) -m pip install -U pip
	$(PYBIN) -m pip install -r requirements.txt

video:
	$(PYBIN) -m src.cli.hip_height_from_video --video "$(VIDEO)" --conf $(CONF) --fps $(FPS) $(DISPLAY_FLAG) --save_conf $(SAVE_CONF) --save_dir "$(SAVE_DIR)" --output "$(OUTPUT)" $(CUDA_FLAG)

cam:
	$(PYBIN) -m src.cli.hip_height_from_video --video $(CAM) --conf $(CONF) --fps $(FPS) $(DISPLAY_FLAG) $(CUDA_FLAG)

image:
	$(PYBIN) -m src.cli.hip_height_from_image --img "$(IMG)" --conf $(CONF) $(METHOD_FLAG)

# todo: make these once gui and server have been finalised
# start_server:

start_client: # todo: impove
	python src/gui/server_client/client_gui.py


clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache outputs/* *.log output.mp4

# ---- macOS App Building ----
build-apps:
	@echo "Building macOS .app bundles..."
	@chmod +x build_apps.sh
	@./build_apps.sh

build-demo:
	@echo "Building HumanHeightDemo.app..."
	@python3 -m PyInstaller demo.spec --clean --noconfirm

build-depth:
	@echo "Building DepthHumanHeight.app..."
	@python3 -m PyInstaller depth_human_height.spec --clean --noconfirm

quick-build:
	@echo "Quick build of both apps..."
	@chmod +x quick_build.sh
	@./quick_build.sh

install-deps:
	@echo "Installing dependencies for app building..."
	@pip3 install -r requirements.txt

clean-apps:
	@echo "Cleaning app build artifacts..."
	@rm -rf build/ dist/ *.spec.bak
