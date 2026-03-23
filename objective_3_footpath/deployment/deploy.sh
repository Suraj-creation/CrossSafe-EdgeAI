#!/bin/bash
# ============================================================================
# Objective 3 — Deployment Script for Raspberry Pi 4 / Jetson Nano
# Run this on the edge device after copying the project files.
# ============================================================================

set -e

INSTALL_DIR="/home/pi/objective_3_footpath"
SERVICE_NAME="pedestrian_ai_obj3"

echo "========================================="
echo "Objective 3 — Edge Deployment"
echo "========================================="

# 1. Install system dependencies
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev \
    mosquitto-clients

# 2. Create virtual environment
echo "[2/6] Setting up Python environment..."
python3 -m venv "${INSTALL_DIR}/venv"
source "${INSTALL_DIR}/venv/bin/activate"

# 3. Install Python dependencies
echo "[3/6] Installing Python packages..."
pip install --upgrade pip
pip install -r "${INSTALL_DIR}/requirements_edge.txt"

# 4. Verify models exist
echo "[4/6] Checking models..."
MODELS_OK=true
for model in twowheeler_int8.tflite lp_localise_int8.tflite; do
    if [ ! -f "${INSTALL_DIR}/models/${model}" ]; then
        echo "  [WARN] Missing: models/${model}"
        MODELS_OK=false
    else
        echo "  [OK]   models/${model}"
    fi
done

# 5. Install systemd service
echo "[5/6] Installing systemd service..."
sudo cp "${INSTALL_DIR}/deployment/${SERVICE_NAME}.service" \
    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

# 6. Test run
echo "[6/6] Running test (5 seconds)..."
timeout 5 python3 "${INSTALL_DIR}/main.py" --source 0 || true

echo ""
echo "========================================="
echo "Deployment complete!"
echo "========================================="
echo "  Start:  sudo systemctl start ${SERVICE_NAME}"
echo "  Status: sudo systemctl status ${SERVICE_NAME}"
echo "  Logs:   journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  Before first run, calibrate the camera:"
echo "    python3 scripts/calibration_tool.py --source 0"
