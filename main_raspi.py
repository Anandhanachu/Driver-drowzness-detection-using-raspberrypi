"""
Driver Drowsiness Detection — Raspberry Pi Edition
====================================================
Detects driver drowsiness using Eye Aspect Ratio (EAR) from an IP camera.

Output (GPIO only — no display window):
  • GPIO 17 → LED blinks (4 Hz) while drowsy        [gpiozero LED]
  • GPIO 27 → Passive buzzer sounds while drowsy     [gpiozero PWMOutputDevice]

Video source: IP camera MJPEG stream → http://192.0.0.2:8081

Requirements:
    pip install -r requirements_raspi.txt

Download shape predictor (run once):
    wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

Hardware wiring:
    LED   (+resistor 220Ω) → GPIO 17  (Physical Pin 11)
    Passive Buzzer (+)     → GPIO 27  (Physical Pin 13)
    Both GND               → GND      (Physical Pin 6 or 9)
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
CAMERA_URL       = "http://192.0.0.2:8081"               # IP camera MJPEG stream
PREDICTOR_PATH   = "shape_predictor_68_face_landmarks.dat"

PIN_LED          = 17       # GPIO BCM pin → LED
PIN_BUZZER       = 27       # GPIO BCM pin → Passive Buzzer (PWM)
LED_BLINK_HZ     = 4        # LED blink rate when drowsy (Hz)
BUZZER_FREQ      = 1000     # Buzzer tone frequency in Hz (1 kHz)
BUZZER_DUTY      = 0.5      # Duty cycle 0.0–1.0  (0.5 = 50%, loudest for passive buzzer)

EAR_CLOSED_RATIO = 0.75     # EAR threshold = baseline_ear × this ratio
DROWSY_SECONDS   = 0.7      # Eyes must be closed this long before alert fires
ALERT_RESEND_SEC = 5        # Heartbeat: re-assert GPIO every N seconds while drowsy
CALIBRATION_SECS = 3        # Seconds to sample open-eye EAR baseline at startup


# ─────────────────────────────────────────────
# GPIO SETUP  (gpiozero)
# ─────────────────────────────────────────────
try:
    from gpiozero import LED as GpioLED, PWMOutputDevice
    from signal import pause

    led    = GpioLED(PIN_LED)
    buzzer = PWMOutputDevice(PIN_BUZZER, frequency=BUZZER_FREQ)
    buzzer.value = 0          # start silent

    gpio_available = True
    print(f"[GPIO] Initialised — LED=GPIO{PIN_LED}, Buzzer=GPIO{PIN_BUZZER} (PWM {BUZZER_FREQ} Hz)")

except ImportError:
    print("[GPIO] WARNING: gpiozero not installed. Running without GPIO output.")
    gpio_available = False
except Exception as e:
    print(f"[GPIO] WARNING: GPIO setup failed: {e}. Running without GPIO output.")
    gpio_available = False


# ─────────────────────────────────────────────
# GPIO HELPERS
# ─────────────────────────────────────────────
def gpio_alert_on():
    """Start LED blinking + passive buzzer tone."""
    if gpio_available:
        led.blink(on_time=1/LED_BLINK_HZ, off_time=1/LED_BLINK_HZ)  # 4 Hz blink
        buzzer.value = BUZZER_DUTY                                    # PWM → buzzer sounds
    print("[GPIO] ALERT ON  — LED blinking, Buzzer sounding")


def gpio_alert_off():
    """Stop LED + silence passive buzzer."""
    if gpio_available:
        led.off()
        buzzer.value = 0      # duty cycle 0 → buzzer silent
    print("[GPIO] ALERT OFF — LED off, Buzzer silent")


def gpio_cleanup():
    """Turn off all GPIO devices on exit."""
    if gpio_available:
        led.off()
        buzzer.value = 0
        buzzer.close()
        led.close()
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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # keep buffer minimal to reduce latency

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
EAR_THRESHOLD = 0.22            # safe fallback if no face detected during calibration

while time.time() - calib_start < CALIBRATION_SECS:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape_np = face_utils.shape_to_np(predictor(gray, face))
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

print("[System] Running. Output: LED (GPIO 17) + Buzzer (GPIO 27 PWM). Ctrl+C to quit.\n")


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
