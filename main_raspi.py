"""
Driver Drowsiness Detection — Raspberry Pi Edition
====================================================
Detects driver drowsiness using Eye Aspect Ratio (EAR) from an IP camera.

Output:
  • GPIO 17 → LED blinks (4 Hz) while drowsy
  • GPIO 27 → Active buzzer ON while drowsy

No display window — output is ONLY via GPIO (LED + Buzzer).

Video source: IP camera MJPEG stream → http://192.0.0.2:8081

Requirements:
    pip install -r requirements_raspi.txt

Download shape predictor (run once):
    wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

Hardware wiring:
    LED  (+resistor 220Ω) → GPIO 17  (Physical Pin 11)
    Active Buzzer         → GPIO 27  (Physical Pin 13)
    Both GND              → GND      (Physical Pin 6 or 9)
"""

import cv2
import dlib
import time
import threading
import numpy as np
from scipy.spatial import distance
from imutils import face_utils

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CAMERA_URL       = "http://192.0.0.2:8081"              # IP camera MJPEG stream
PREDICTOR_PATH   = "shape_predictor_68_face_landmarks.dat"

PIN_LED          = 17       # GPIO BCM pin → LED
PIN_BUZZER       = 27       # GPIO BCM pin → Active Buzzer
LED_BLINK_HZ     = 4        # LED blink rate when drowsy (Hz)

EAR_CLOSED_RATIO = 0.75     # EAR threshold = baseline_ear × this ratio
DROWSY_SECONDS   = 0.7      # Eyes must be closed for this long before alert fires
ALERT_RESEND_SEC = 5        # Heartbeat: re-assert GPIO every N seconds while drowsy
CALIBRATION_SECS = 3        # Seconds to sample open-eye EAR at startup


# ─────────────────────────────────────────────
# GPIO SETUP
# ─────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_LED,    GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.LOW)
    gpio_available = True
    print(f"[GPIO] Initialised — LED=GPIO{PIN_LED}, BUZZER=GPIO{PIN_BUZZER}")
except ImportError:
    print("[GPIO] WARNING: RPi.GPIO not installed. Running without GPIO output.")
    gpio_available = False
except Exception as e:
    print(f"[GPIO] WARNING: GPIO setup failed: {e}. Running without GPIO output.")
    gpio_available = False


# ─────────────────────────────────────────────
# LED BLINK THREAD
# ─────────────────────────────────────────────
_blink_event  = threading.Event()
_blink_thread = None

def _blink_worker():
    """Background thread: blink LED at LED_BLINK_HZ until _blink_event is set."""
    half_period = 1.0 / (LED_BLINK_HZ * 2)
    while not _blink_event.is_set():
        if gpio_available:
            GPIO.output(PIN_LED, GPIO.HIGH)
        _blink_event.wait(timeout=half_period)
        if gpio_available:
            GPIO.output(PIN_LED, GPIO.LOW)
        _blink_event.wait(timeout=half_period)
    if gpio_available:
        GPIO.output(PIN_LED, GPIO.LOW)     # ensure off on thread exit


def gpio_alert_on():
    """Activate alert: start LED blink thread + turn buzzer ON."""
    global _blink_thread
    if _blink_thread is None or not _blink_thread.is_alive():
        _blink_event.clear()
        _blink_thread = threading.Thread(target=_blink_worker, daemon=True)
        _blink_thread.start()
    if gpio_available:
        GPIO.output(PIN_BUZZER, GPIO.HIGH)
    print("[GPIO] ALERT ON  — LED blinking, Buzzer HIGH")


def gpio_alert_off():
    """Deactivate alert: stop LED + turn buzzer OFF."""
    _blink_event.set()
    if gpio_available:
        GPIO.output(PIN_LED,    GPIO.LOW)
        GPIO.output(PIN_BUZZER, GPIO.LOW)
    print("[GPIO] ALERT OFF — LED off, Buzzer LOW")


def gpio_cleanup():
    """Safe GPIO cleanup on exit."""
    _blink_event.set()
    if gpio_available:
        GPIO.output(PIN_LED,    GPIO.LOW)
        GPIO.output(PIN_BUZZER, GPIO.LOW)
        GPIO.cleanup()
    print("[GPIO] Cleanup done.")


# ─────────────────────────────────────────────
# EAR HELPER
# ─────────────────────────────────────────────
def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Return Eye Aspect Ratio (EAR) from 6 eye landmark points."""
    if len(eye) < 6:
        return 0.0
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ─────────────────────────────────────────────
# DLIB SETUP
# ─────────────────────────────────────────────
print("[dlib] Loading face detector and shape predictor...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("[dlib] Ready.")


# ─────────────────────────────────────────────
# IP CAMERA SETUP
# ─────────────────────────────────────────────
print(f"[Camera] Connecting to: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # keep buffer small to minimise latency

for attempt in range(10):
    if cap.isOpened():
        break
    print(f"[Camera] Waiting for stream... attempt {attempt + 1}/10")
    time.sleep(2)
    cap.open(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot connect to IP camera at {CAMERA_URL}.\n"
        "Ensure the camera is on the same network and the URL is correct."
    )

ret, test_frame = cap.read()
if not ret or test_frame is None:
    raise RuntimeError("Stream opened but cannot read frame. Check camera format.")

frame_h, frame_w = test_frame.shape[:2]
print(f"[Camera] Stream open — {frame_w}x{frame_h}")


# ─────────────────────────────────────────────
# CALIBRATION — measure baseline open-eye EAR
# ─────────────────────────────────────────────
print(f"[Calibration] Keep your eyes OPEN for {CALIBRATION_SECS} seconds...")
ear_samples   = []
calib_start   = time.time()
EAR_THRESHOLD = 0.22           # safe fallback if no face detected

while time.time() - calib_start < CALIBRATION_SECS:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape    = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)
        ear = (eye_aspect_ratio(shape_np[lStart:lEnd]) +
               eye_aspect_ratio(shape_np[rStart:rEnd])) / 2.0
        ear_samples.append(ear)
        break   # only first face needed

if ear_samples:
    baseline_ear  = np.mean(ear_samples)
    EAR_THRESHOLD = baseline_ear * EAR_CLOSED_RATIO
    print(f"[Calibration] Baseline EAR: {baseline_ear:.3f}  →  Threshold: {EAR_THRESHOLD:.3f}")
else:
    print(f"[Calibration] No face detected. Using default threshold: {EAR_THRESHOLD:.3f}")


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
sleep_state       = False
eyes_closed_since = None
last_alert_ts     = 0.0

print("[System] Running. Output: LED (GPIO 17) + Buzzer (GPIO 27). Ctrl+C to quit.\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Camera] Frame read failed — retrying...")
            time.sleep(0.1)
            cap.open(CAMERA_URL)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        now   = time.time()

        # ── NO FACE ──────────────────────────────────
        if len(faces) == 0:
            eyes_closed_since = None
            if sleep_state:
                gpio_alert_off()
                sleep_state = False
                print("[Alert] Face lost — alert cleared.")
            continue

        # ── FIRST FACE ───────────────────────────────
        face     = faces[0]
        shape_np = face_utils.shape_to_np(predictor(gray, face))
        ear      = (eye_aspect_ratio(shape_np[lStart:lEnd]) +
                    eye_aspect_ratio(shape_np[rStart:rEnd])) / 2.0

        # ── DROWSINESS DECISION ──────────────────────
        if ear < EAR_THRESHOLD:
            if eyes_closed_since is None:
                eyes_closed_since = now
            elapsed = now - eyes_closed_since

            if elapsed >= DROWSY_SECONDS:
                if not sleep_state:
                    gpio_alert_on()
                    last_alert_ts = now
                    sleep_state   = True
                    print(f"[Alert] DROWSY — EAR={ear:.3f} < {EAR_THRESHOLD:.3f}")
                elif now - last_alert_ts >= ALERT_RESEND_SEC:
                    gpio_alert_on()          # heartbeat: keep GPIO asserted
                    last_alert_ts = now
                    print(f"[Alert] DROWSY (heartbeat) — EAR={ear:.3f}")
        else:
            eyes_closed_since = None
            if sleep_state:
                gpio_alert_off()
                sleep_state = False
                print(f"[Alert] AWAKE  — EAR={ear:.3f}")

except KeyboardInterrupt:
    print("\n[System] Stopped by user.")

finally:
    if sleep_state:
        gpio_alert_off()
    gpio_cleanup()
    cap.release()
    print("[System] Done.")
