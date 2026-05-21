"""
Drowsiness Detection System
============================
Uses facial landmark EAR (Eye Aspect Ratio) + head pose pitch
to detect driver drowsiness and alert via Arduino serial.

Requirements:
    pip install opencv-python dlib imutils scipy pyserial numpy

Download shape predictor:
    https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
"""

import cv2
import dlib
import serial
import time
import numpy as np
from scipy.spatial import distance
from imutils import face_utils

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SERIAL_PORT       = "COM2"       # Change to your Arduino port (e.g. /dev/ttyUSB0 on Linux)
BAUD_RATE         = 9600
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"

EAR_CLOSED_RATIO  = 0.75          # Threshold = baseline_ear * this value (calibration)
DROWSY_SECONDS    = 0.7           # Seconds eyes must be closed to trigger alert
ALERT_RESEND_SEC  = 5             # Re-send 'D' every N seconds while still drowsy
CALIBRATION_SECS  = 3             # Seconds to calibrate open-eye EAR at startup
BLINK_IGNORE_SEC  = 0.15          # Fast blinks under this duration are ignored

# 3D model points for head pose (generic face model, millimetres)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip (30)
    (0.0,   -330.0, -65.0),    # Chin (8)
    (-225.0, 170.0, -135.0),   # Left eye left corner (36)
    (225.0,  170.0, -135.0),   # Right eye right corner (45)
    (-150.0, -150.0, -125.0),  # Left mouth corner (48)
    (150.0,  -150.0, -125.0),  # Right mouth corner (54)
], dtype=np.float64)

LANDMARK_IDS = [30, 8, 36, 45, 48, 54]   # Corresponding 68-landmark indices

# Head pose drowsiness: pitch > this angle (degrees) = head drooping
HEAD_PITCH_THRESHOLD = 20.0

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def eye_aspect_ratio(eye: np.ndarray) -> float:
    if len(eye) < 6:          # ← add this guard
        return 0.0
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pitch(shape: np.ndarray, frame_w: int, frame_h: int) -> float | None:
    """
    Estimate head pitch (up/down tilt) using solvePnP.
    Returns pitch in degrees, or None if estimation fails.
    """
    image_points = np.array(
        [shape[i] for i in LANDMARK_IDS], dtype=np.float64
    )
    focal_length = frame_w
    center       = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    # Decompose to Euler angles (pitch = rotation around X axis)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
    return pitch


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


# ─────────────────────────────────────────────
# SERIAL SETUP
# ─────────────────────────────────────────────
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[Serial] Connected on {SERIAL_PORT}")
    serial_ok = True
except serial.SerialException as e:
    print(f"[Serial] WARNING: Could not open {SERIAL_PORT}: {e}")
    print("[Serial] Continuing without serial output.")
    ser = None
    serial_ok = False


def serial_send(byte: bytes):
    """Send a byte over serial if the connection is active."""
    if serial_ok and ser and ser.is_open:
        try:
            ser.write(byte)
        except serial.SerialException as e:
            print(f"[Serial] Write error: {e}")


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
# CAMERA SETUP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found. Check device index.")

ret, test_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera.")

frame_h, frame_w = test_frame.shape[:2]
print(f"[Camera] Resolution: {frame_w}x{frame_h}")

# ─────────────────────────────────────────────
# CALIBRATION PHASE
# ─────────────────────────────────────────────
print(f"[Calibration] Keep eyes open for {CALIBRATION_SECS} seconds...")
ear_samples  = []
calib_start  = time.time()
EAR_THRESHOLD = 0.22  # safe fallback default

while time.time() - calib_start < CALIBRATION_SECS:
    ret, frame = cap.read()
    if not ret:
        continue
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    remaining = int(CALIBRATION_SECS - (time.time() - calib_start)) + 1
    draw_text(frame, "CALIBRATING — KEEP EYES OPEN", (30, 40),  (0, 220, 255), scale=0.75)
    draw_text(frame, f"Please wait: {remaining}s",   (30, 75),  (255, 255, 255), scale=0.65)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        ear_samples.append(ear)
        draw_text(frame, f"EAR: {ear:.3f}", (30, 110), (200, 200, 200), scale=0.6)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        exit()

if ear_samples:
    baseline_ear  = np.mean(ear_samples)
    EAR_THRESHOLD = baseline_ear * EAR_CLOSED_RATIO
    print(f"[Calibration] Baseline EAR: {baseline_ear:.3f}  →  Threshold set to: {EAR_THRESHOLD:.3f}")
else:
    print(f"[Calibration] No face detected. Using default threshold: {EAR_THRESHOLD}")

# ─────────────────────────────────────────────
# STATE VARIABLES
# ─────────────────────────────────────────────
sleep_state       = False         # True while drowsy alert is active
eyes_closed_since = None          # Timestamp when eyes first closed
last_alert_sent   = 0.0           # Timestamp of last 'D' serial send
alert_reason      = ""            # "EAR" or "HEAD"

# FPS tracking
fps_counter = 0
fps_start   = time.time()
fps_display = 0.0

print("[System] Drowsiness detection running. Press 'q' to quit.\n")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed. Retrying...")
            time.sleep(0.05)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        # ── FPS ──────────────────────────────
        fps_counter += 1
        if fps_counter >= 10:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        now         = time.time()
        ear         = 0.0
        pitch       = None
        face_found  = len(faces) > 0

        # ── NO FACE ──────────────────────────
        if not face_found:
            # Reset closed timer but keep sleep_state until face returns,
            # then clear it so Arduino resets.
            eyes_closed_since = None
            if sleep_state:
                serial_send(b'A')
                print("[Alert] Face lost — reset sent.")
                sleep_state  = False
                alert_reason = ""
            draw_text(frame, "NO FACE DETECTED", (30, 50), (0, 165, 255))

        # ── FACE FOUND ───────────────────────
        for face in faces:
            shape    = predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            leftEye  = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR  = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear      = (leftEAR + rightEAR) / 2.0

            # Head pose pitch
            pitch = get_head_pitch(shape_np, frame_w, frame_h)

            # Draw eye contours
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 100, 255)
            draw_eye_contour(frame, leftEye,  eye_color)
            draw_eye_contour(frame, rightEye, eye_color)

            # Draw bounding box
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)

            # ── DROWSINESS DETECTION ─────────
            ear_drowsy  = ear < EAR_THRESHOLD
            head_drowsy = (pitch is not None) and (pitch < -HEAD_PITCH_THRESHOLD)
            is_drowsy   = ear_drowsy or head_drowsy

            if is_drowsy:
                if eyes_closed_since is None:
                    eyes_closed_since = now             # start timer
                elapsed = now - eyes_closed_since

                if elapsed >= DROWSY_SECONDS:
                    # Trigger or sustain alert
                    if not sleep_state:
                        serial_send(b'D')
                        last_alert_sent = now
                        sleep_state     = True
                        alert_reason    = "EAR" if ear_drowsy else "HEAD"
                        print(f"[Alert] DROWSY ({alert_reason}) — 'D' sent.")
                    elif now - last_alert_sent >= ALERT_RESEND_SEC:
                        serial_send(b'D')               # heartbeat resend
                        last_alert_sent = now
                        print(f"[Alert] DROWSY heartbeat — 'D' resent.")
                else:
                    # Eyes closed but timer not yet reached
                    bar_w   = int((elapsed / DROWSY_SECONDS) * 200)
                    bar_col = (0, int(255 * (1 - elapsed / DROWSY_SECONDS)), int(255 * elapsed / DROWSY_SECONDS))
                    cv2.rectangle(frame, (30, frame_h - 30), (30 + bar_w, frame_h - 15), bar_col, -1)
                    cv2.rectangle(frame, (30, frame_h - 30), (230,        frame_h - 15), (200, 200, 200), 1)
                    draw_text(frame, "Checking...", (240, frame_h - 16), (255, 255, 0), scale=0.5)

            else:
                # Eyes open and head up
                eyes_closed_since = None
                if sleep_state:
                    serial_send(b'A')
                    print("[Alert] AWAKE — 'A' sent.")
                    sleep_state  = False
                    alert_reason = ""

            break  # process only the first/largest face

        # ── HUD OVERLAY ──────────────────────
        # Status banner
        if sleep_state:
            reason_str = f"DROWSY ({alert_reason})"
            cv2.rectangle(frame, (0, 0), (frame_w, 70), (0, 0, 180), -1)
            draw_text(frame, reason_str, (30, 48), (255, 255, 255), scale=1.1, thickness=3)
        elif face_found:
            cv2.rectangle(frame, (0, 0), (frame_w, 70), (0, 120, 0), -1)
            draw_text(frame, "AWAKE", (30, 48), (255, 255, 255), scale=1.1, thickness=3)

        # Stats panel
        if face_found:
            stats = [
                f"EAR   : {ear:.3f}  (threshold {EAR_THRESHOLD:.3f})",
                f"Pitch : {pitch:.1f} deg" if pitch is not None else "Pitch : N/A",
                f"FPS   : {fps_display:.1f}",
            ]
            panel_y = 85
            for line in stats:
                draw_text(frame, line, (10, panel_y), (220, 220, 220), scale=0.55, thickness=1)
                panel_y += 22

        # Serial status indicator
        status_color = (0, 200, 0) if serial_ok else (0, 80, 255)
        status_label = f"Serial: {SERIAL_PORT}" if serial_ok else "Serial: OFFLINE"
        draw_text(frame, status_label, (frame_w - 260, frame_h - 12), status_color, scale=0.5, thickness=1)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[System] Quit requested.")
            break

finally:
    # ── CLEANUP ──────────────────────────────
    print("[System] Shutting down...")
    if sleep_state and serial_ok:
        serial_send(b'A')                  # ensure Arduino resets on exit
    cap.release()
    if ser and ser.is_open:
        ser.close()
        print("[Serial] Port closed.")
    cv2.destroyAllWindows()
    print("[System] Done.")