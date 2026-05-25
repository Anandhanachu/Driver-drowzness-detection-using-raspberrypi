# Driver Drowsiness Detection — Raspberry Pi Edition

> Camera-based driver drowsiness detection that runs **entirely on a Raspberry Pi**.  
> Detects closed eyes using the **Eye Aspect Ratio (EAR)** — alerts the driver through a **blinking LED** and an **active buzzer**.

---

## Table of Contents
1. [How It Works](#how-it-works)
2. [Hardware Requirements](#hardware-requirements)
3. [GPIO Wiring](#gpio-wiring)
4. [Software Setup](#software-setup)
   - [1. OS & System Packages](#1-os--system-packages)
   - [2. Python Virtual Environment](#2-python-virtual-environment)
   - [3. Install dlib (compiled on RPi)](#3-install-dlib-compiled-on-rpi)
   - [4. Install Remaining Dependencies](#4-install-remaining-dependencies)
   - [5. Download the Shape Predictor File](#5-download-the-shape-predictor-file)
5. [IP Camera Setup](#ip-camera-setup)
6. [Configuration](#configuration)
7. [Running the Script](#running-the-script)
8. [Run on Boot (Systemd)](#run-on-boot-systemd)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## How It Works

```
IP Camera (http://192.0.0.2:8081)
         │
         ▼
  OpenCV VideoCapture
         │
         ▼
  dlib face detector
         │
         ▼
  68-point facial landmarks
         │
  EAR (Eye Aspect Ratio)
  eyes closed → EAR drops below threshold
         │
  Timer: drowsy for > 0.7 s?
         │  YES
         ▼
  GPIO 17 → LED blinks (4 Hz)
  GPIO 27 → Active Buzzer ON
         │
         ▼
  Driver wakes up → GPIO OFF
```

**Detection method:**
| Method | Description |
|---|---|
| EAR (Eye Aspect Ratio) | Ratio of eye height to width. Falls below threshold when eyes close. Auto-calibrated at startup. |

---

## Hardware Requirements

| Component | Qty | Notes |
|---|---|---|
| Raspberry Pi (3B / 4 / Zero 2W) | 1 | RPi 4 recommended for dlib speed |
| MicroSD (≥16 GB, Class 10) | 1 | Raspberry Pi OS (64-bit recommended) |
| IP Camera | 1 | Any camera that streams MJPEG over HTTP |
| LED (any colour) | 1 | Standard 5mm LED |
| Resistor 220Ω | 1 | Current limiting for LED |
| Active Buzzer | 1 | **Active** type (not passive — no PWM needed) |
| Jumper wires | several | |
| Breadboard | 1 | Optional, for prototyping |

> **Active vs Passive Buzzer**: An *active* buzzer makes sound when you apply DC HIGH.  
> A *passive* buzzer requires a PWM signal. This project uses an **active** buzzer.

---

## GPIO Wiring

```
Raspberry Pi                   Components
─────────────                  ──────────────────────────────
Physical Pin 11  (GPIO 17) ───►  [220Ω resistor] ──► LED (+)
Physical Pin 13  (GPIO 27) ───►  Active Buzzer (+)
Physical Pin 6   (GND)     ───►  LED (−) + Buzzer (−) [common GND]
```

### Pinout Diagram

```
     RPi Header (top view, odd pins left)
     ┌─────────────────────┐
  3V3│ 1   2 │5V
 SDA1│ 3   4 │5V
 SCL1│ 5   6 │GND ◄── common ground (LED & buzzer)
GPIO4│ 7   8 │TXD
  GND│ 9  10 │RXD
GPIO17│11  12│GPIO18   ◄── Pin 11 = LED
GPIO27│13  14│GND      ◄── Pin 13 = Buzzer
     └─────────────────────┘
```

---

## Software Setup

### 1. OS & System Packages

Flash **Raspberry Pi OS (Bookworm, 64-bit)** to your SD card, then:

```bash
sudo apt update && sudo apt upgrade -y

# Build tools (needed for dlib)
sudo apt install -y build-essential cmake pkg-config git

# Python dev headers
sudo apt install -y python3-dev python3-pip python3-venv

# OpenCV system dependencies
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
    libopenblas-dev libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev

# Increase swap space (required for compiling dlib on RPi 3/Zero)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 2. Python Virtual Environment

```bash
cd ~
python3 -m venv drowsy-env
source drowsy-env/bin/activate
pip install --upgrade pip wheel setuptools
```

### 3. Install dlib (compiled on RPi)

> ⚠️ This step takes **20–40 minutes** on RPi 3/4. Use all 4 CPU cores to speed it up.

```bash
# Option A: Compile from source (most compatible)
pip install dlib --verbose

# If the above fails due to memory, set cmake args explicitly:
export CMAKE_BUILD_PARALLEL_LEVEL=4
pip install dlib --no-cache-dir
```

> **Tip for RPi 4 (8GB)**: Compilation is ~15 min. For RPi 3/Zero, ensure swap is increased (step 1).

### 4. Install Remaining Dependencies

```bash
pip install -r requirements_raspi.txt
```

> `RPi.GPIO` is already included in Raspberry Pi OS, but installing via pip ensures the correct version.

### 5. Download the Shape Predictor File

The `shape_predictor_68_face_landmarks.dat` file is **required** but not included in this repository (it is ~100 MB).

**Option A — wget (on the RPi):**
```bash
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

**Option B — curl:**
```bash
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
    https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

**Option C — Transfer from PC (if already downloaded):**
```bash
# Run this from your PC/laptop
scp shape_predictor_68_face_landmarks.dat pi@<raspi-ip>:~/driver-drowsiness-detection/
```

Place the `.dat` file in the **same directory** as `main_raspi.py`.

---

## IP Camera Setup

This project reads video from an **IP camera MJPEG stream**.

| Setting | Value |
|---|---|
| Default URL | `http://192.0.0.2:8081` |
| Protocol | HTTP MJPEG |

**Ensure the RPi can reach the camera:**
```bash
ping 192.0.0.2
curl -I http://192.0.0.2:8081    # should return HTTP 200
```

**If your camera uses a different URL format**, edit `CAMERA_URL` in `main_raspi.py`:
```python
CAMERA_URL = "http://192.0.0.2:8081"   # MJPEG stream
# CAMERA_URL = "rtsp://user:pass@192.0.0.2:554/stream"  # RTSP example
```

**IP Camera apps for smartphones** (for testing):
- **DroidCam** (Android/iOS) — set URL to `http://<phone-ip>:4747/video`
- **IP Webcam** (Android) — set URL to `http://<phone-ip>:8080/video`

---

## Configuration

All tunable parameters are at the top of `main_raspi.py`:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_URL` | `http://192.0.0.2:8081` | IP camera MJPEG stream URL |
| `PIN_LED` | `17` | GPIO BCM pin number for LED |
| `PIN_BUZZER` | `27` | GPIO BCM pin number for buzzer |
| `LED_BLINK_HZ` | `4` | LED blink rate when drowsy (Hz) |
| `EAR_CLOSED_RATIO` | `0.75` | EAR threshold = baseline × this ratio |
| `DROWSY_SECONDS` | `0.7` | How long eyes must be closed before alert |
| `CALIBRATION_SECS` | `3` | Calibration duration at startup |
| `HEADLESS` | `True` | `True` = no display window (SSH-safe) |

---

## Running the Script

```bash
# Activate virtual environment
source ~/drowsy-env/bin/activate

# Navigate to project directory
cd ~/driver-drowsiness-detection

# Run
python main_raspi.py
```

**Expected startup output:**
```
[GPIO] Initialised — LED=GPIO17, BUZZER=GPIO27
[dlib] Loading face detector and shape predictor...
[dlib] Ready.
[Camera] Connecting to IP stream: http://192.0.0.2:8081
[Camera] Stream open — resolution: 640x480
[Calibration] Keep eyes open for 3 seconds...
[Calibration] Baseline EAR: 0.312  →  Threshold: 0.234
[System] Drowsiness detection running. Press Ctrl+C to quit.
```

**Stop with:** `Ctrl + C`

---

## Run on Boot (Systemd)

To start the detection automatically when the RPi boots:

```bash
sudo nano /etc/systemd/system/drowsy.service
```

Paste:
```ini
[Unit]
Description=Driver Drowsiness Detection
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/driver-drowsiness-detection
ExecStart=/home/pi/drowsy-env/bin/python main_raspi.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable drowsy.service
sudo systemctl start drowsy.service

# Check status
sudo systemctl status drowsy.service

# View logs
journalctl -u drowsy.service -f
```

---

## Troubleshooting

### Camera not connecting
```
RuntimeError: Cannot connect to IP camera at http://192.0.0.2:8081
```
- Verify the RPi and camera are on the same network: `ping 192.0.0.2`
- Test the URL in a browser on the RPi
- Some cameras require `/mjpg/video.mjpg` or `/video` as the path

### dlib fails to compile
- Ensure swap is increased (see step 1)
- Install `cmake` first: `sudo apt install cmake`
- Try: `pip install dlib --no-cache-dir`

### GPIO permission denied
```bash
sudo usermod -aG gpio $USER
# Log out and back in
```
Or run with `sudo python main_raspi.py` temporarily.

### shape_predictor file not found
```
RuntimeError: Unable to open shape_predictor_68_face_landmarks.dat
```
Download the file (see [Step 5](#5-download-the-shape-predictor-file)) and place it in the same folder as `main_raspi.py`.

### Low FPS / slow detection
- Reduce resolution: edit `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` in `main_raspi.py`
- Use RPi 4 (significantly faster than RPi 3)
- Ensure only 1 face is in frame (detector is faster with 1 face)
- Upgrade SD card to A2 speed class

### Buzzer makes no sound
- Confirm you have an **active** buzzer (not passive)
- Test pin directly: `python3 -c "import RPi.GPIO as G; G.setmode(G.BCM); G.setup(27,G.OUT); G.output(27,1); import time; time.sleep(2); G.cleanup()"`

---

## Project Structure

```
driver-drowsiness-detection/
├── main_raspi.py                          # Main script — EAR-based, RPi GPIO output
├── requirements_raspi.txt                 # Python dependencies for RPi
├── shape_predictor_68_face_landmarks.dat  # dlib model (download separately — NOT in git)
└── README.md                             # This file
```

> **Note:** `shape_predictor_68_face_landmarks.dat` is listed in `.gitignore` due to its large size (~100 MB). Always download it separately using the instructions above.

---

## License

MIT License — see individual library licenses for dlib, OpenCV, and imutils.
