"""
Driver Drowsiness Detection — Raspberry Pi Edition
====================================================
Uses Eye Aspect Ratio (EAR) to detect driver drowsiness and alerts via:
  • GPIO LED  (blinking at PIN_LED while drowsy)
  • GPIO active buzzer (continuous HIGH at PIN_BUZZER while drowsy)

Video source: IP camera MJPEG stream → http://192.0.0.2:8081

Requirements (see requirements_raspi.txt):
    pip install opencv-python dlib imutils scipy numpy RPi.GPIO

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
CAMERA_URL      = "http://192.0.0.2:8081"   # IP camera MJPEG stream
PREDICTOR_PATH  = "shape_predictor_68_face_landmarks.dat"

PIN_LED         = 17          # GPIO BCM pin for LED
PIN_BUZZER      = 27          # GPIO BCM pin for active buzzer
LED_BLINK_HZ    = 4           # LED blink frequency when drowsy (blinks per second)

EAR_CLOSED_RATIO = 0.75       # EAR threshold = baseline_ear * this ratio (calibration)
DROWSY_SECONDS   = 0.7        # Seconds eyes must stay closed before alert triggers
ALERT_RESEND_SEC = 5          # Heartbeat: re-assert GPIO every N seconds while drowsy
CALIBRATION_SECS = 3          # Seconds to calibrate open-eye EAR baseline at startup

HEADLESS        = True        # True  = no display window (SSH / headless RPi)
                              # False = show video window (monitor connected)


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
    print("[GPIO] WARNING: RPi.GPIO not available. Running without GPIO output.")
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
    period = 1.0 / (LED_BLINK_HZ * 2)     # half-period per toggle
    while not _blink_event.is_set():
        if gpio_available:
            GPIO.output(PIN_LED, GPIO.HIGH)
        _blink_event.wait(timeout=period)
        if gpio_available:
            GPIO.output(PIN_LED, GPIO.LOW)
        _blink_event.wait(timeout=period)
    if gpio_available:
        GPIO.output(PIN_LED, GPIO.LOW)      # ensure LED off on exit


def gpio_alert_on():
    """Activate drowsiness alert: start LED blinking + turn buzzer ON."""
    global _blink_thread
    if _blink_thread is None or not _blink_thread.is_alive():
        _blink_event.clear()
        _blink_thread = threading.Thread(target=_blink_worker, daemon=True)
        _blink_thread.start()
    if gpio_available:
        GPIO.output(PIN_BUZZER, GPIO.HIGH)
    print("[GPIO] ALERT ON  — LED blinking, Buzzer HIGH")


def gpio_alert_off():
    """Deactivate alert: stop LED blinking + turn buzzer OFF."""
    _blink_event.set()
    if gpio_available:
        GPIO.output(PIN_LED,    GPIO.LOW)
        GPIO.output(PIN_BUZZER, GPIO.LOW)
    print("[GPIO] ALERT OFF — LED off, Buzzer LOW")


def gpio_cleanup():
    """Clean up GPIO pins on exit."""
    _blink_event.set()
    if gpio_available:
        GPIO.output(PIN_LED,    GPIO.LOW)
        GPIO.output(PIN_BUZZER, GPIO.LOW)
        GPIO.cleanup()
    print("[GPIO] Cleanup done.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmark points."""
    if len(eye) < 6:
        return 0.0
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def draw_text(frame, text, pos, color, scale=0.8, thickness=2):
    """Draw text with a dark shadow for readability."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_eye_contour(frame, eye, color):
    hull = cv2.convexHull(eye)
    cv2.drawContours(frame, [hull], -1, color, 1)


def show_frame(frame):
    """Display frame only when not running headless."""
    if not HEADLESS:
        cv2.imshow("Drowsiness Detection [RPi]", frame)


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
print(f"[Camera] Connecting to IP stream: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # minimise latency

for attempt in range(10):
    if cap.isOpened():
        break
    print(f"[Camera] Waiting for stream... attempt {attempt + 1}/10")
    time.sleep(2)
    cap.open(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot connect to IP camera at {CAMERA_URL}.\n"
        "Check that the camera is reachable and the URL is correct."
    )

ret, test_frame = cap.read()
if not ret or test_frame is None:
    raise RuntimeError("Connected but cannot read frame. Check stream format.")

frame_h, frame_w = test_frame.shape[:2]
print(f"[Camera] Stream open — resolution: {frame_w}x{frame_h}")


# ─────────────────────────────────────────────
# CALIBRATION PHASE
# ─────────────────────────────────────────────
print(f"[Calibration] Keep eyes open for {CALIBRATION_SECS} seconds...")
ear_samples   = []
calib_start   = time.time()
EAR_THRESHOLD = 0.22            # fallback if no face detected during calibration

while time.time() - calib_start < CALIBRATION_SECS:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    remaining = int(CALIBRATION_SECS - (time.time() - calib_start)) + 1
    draw_text(frame, "CALIBRATING — KEEP EYES OPEN", (30, 40),  (0, 220, 255), scale=0.75)
    draw_text(frame, f"Please wait: {remaining}s",   (30, 75),  (255, 255, 255), scale=0.65)

    for face in faces:
        shape    = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)
        leftEye  = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        ear_samples.append(ear)
        draw_text(frame, f"EAR: {ear:.3f}", (30, 110), (200, 200, 200), scale=0.6)

    show_frame(frame)
    if not HEADLESS and (cv2.waitKey(1) & 0xFF == ord('q')):
        cap.release()
        gpio_cleanup()
        cv2.destroyAllWindows()
        exit()

if ear_samples:
    baseline_ear  = np.mean(ear_samples)
    EAR_THRESHOLD = baseline_ear * EAR_CLOSED_RATIO
    print(f"[Calibration] Baseline EAR: {baseline_ear:.3f}  →  Threshold: {EAR_THRESHOLD:.3f}")
else:
    print(f"[Calibration] No face detected. Using default threshold: {EAR_THRESHOLD:.3f}")


# ─────────────────────────────────────────────
# STATE VARIABLES
# ─────────────────────────────────────────────
sleep_state       = False     # True while drowsy alert is active
eyes_closed_since = None      # Timestamp when eyes first closed
last_alert_ts     = 0.0       # Timestamp of last GPIO heartbeat

# FPS tracking
fps_counter = 0
fps_start   = time.time()
fps_display = 0.0

print("[System] Drowsiness detection running. Press Ctrl+C to quit.\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Camera] Frame read failed. Retrying...")
            time.sleep(0.1)
            cap.open(CAMERA_URL)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        # ── FPS ──────────────────────────────────────
        fps_counter += 1
        if fps_counter >= 10:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        now        = time.time()
        ear        = 0.0
        face_found = len(faces) > 0

        # ── NO FACE DETECTED ─────────────────────────
        if not face_found:
            eyes_closed_since = None
            if sleep_state:
                gpio_alert_off()
                sleep_state = False
                print("[Alert] Face lost — alert cleared.")
            draw_text(frame, "NO FACE DETECTED", (30, 50), (0, 165, 255))

        # ── PROCESS FIRST FACE ───────────────────────
        for face in faces:
            shape    = predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            leftEye  = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Draw eye contours — green when open, orange when closed
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 140, 255)
            draw_eye_contour(frame, leftEye,  eye_color)
            draw_eye_contour(frame, rightEye, eye_color)

            # Draw face bounding box
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)

            # ── DROWSINESS LOGIC (EAR only) ──────────
            is_drowsy = ear < EAR_THRESHOLD

            if is_drowsy:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                elapsed = now - eyes_closed_since

                if elapsed >= DROWSY_SECONDS:
                    if not sleep_state:
                        gpio_alert_on()
                        last_alert_ts = now
                        sleep_state   = True
                        print("[Alert] DROWSY — triggered.")
                    elif now - last_alert_ts >= ALERT_RESEND_SEC:
                        gpio_alert_on()              # heartbeat: keep GPIO active
                        last_alert_ts = now
                        print("[Alert] DROWSY — heartbeat.")
                else:
                    # Progress bar while timer counts up
                    bar_w   = int((elapsed / DROWSY_SECONDS) * 200)
                    bar_col = (
                        0,
                        int(255 * (1 - elapsed / DROWSY_SECONDS)),
                        int(255 * elapsed / DROWSY_SECONDS)
                    )
                    cv2.rectangle(frame, (30, frame_h - 30), (30 + bar_w, frame_h - 15), bar_col, -1)
                    cv2.rectangle(frame, (30, frame_h - 30), (230,        frame_h - 15), (200, 200, 200), 1)
                    draw_text(frame, "Checking...", (240, frame_h - 16), (255, 255, 0), scale=0.5)

            else:
                eyes_closed_since = None
                if sleep_state:
                    gpio_alert_off()
                    print("[Alert] AWAKE — alert cleared.")
                    sleep_state = False

            break   # process only the first / largest face

        # ── HUD OVERLAY ──────────────────────────────
        if sleep_state:
            cv2.rectangle(frame, (0, 0), (frame_w, 70), (0, 0, 180), -1)
            draw_text(frame, "DROWSY!", (30, 48), (255, 255, 255), scale=1.1, thickness=3)
        elif face_found:
            cv2.rectangle(frame, (0, 0), (frame_w, 70), (0, 120, 0), -1)
            draw_text(frame, "AWAKE", (30, 48), (255, 255, 255), scale=1.1, thickness=3)

        if face_found:
            stats = [
                f"EAR : {ear:.3f}  (thresh {EAR_THRESHOLD:.3f})",
                f"FPS : {fps_display:.1f}",
            ]
            panel_y = 85
            for line in stats:
                draw_text(frame, line, (10, panel_y), (220, 220, 220), scale=0.55, thickness=1)
                panel_y += 22

        gpio_label = "GPIO: OK" if gpio_available else "GPIO: OFFLINE"
        gpio_color = (0, 200, 0) if gpio_available else (0, 80, 255)
        draw_text(frame, gpio_label, (frame_w - 200, frame_h - 12), gpio_color, scale=0.5, thickness=1)

        show_frame(frame)
        if not HEADLESS and (cv2.waitKey(1) & 0xFF == ord('q')):
            print("[System] Quit requested.")
            break

except KeyboardInterrupt:
    print("\n[System] KeyboardInterrupt — shutting down...")

finally:
    print("[System] Shutting down...")
    if sleep_state:
        gpio_alert_off()
    gpio_cleanup()
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("[System] Done.")
