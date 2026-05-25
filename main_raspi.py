"""
Driver Drowsiness Detection — Raspberry Pi Edition
====================================================
Detects driver drowsiness using Eye Aspect Ratio (EAR) from an IP camera.

Output:
  • GPIO 17 → LED blinks (4 Hz) while drowsy        [gpiozero LED]
  • GPIO 27 → Passive buzzer sounds while drowsy     [gpiozero PWMOutputDevice]
  • Gmail   → Emergency email with GPS location      [smtplib — built-in]

Escalation:
  0.7s drowsy  →  LED blink + buzzer starts
  6s  drowsy   →  Emergency email sent (with Google Maps link)
  Driver wakes →  Follow-up "DRIVER SAFE" email sent

Location:
  Primary  → NEO-6M GPS module on GPS_PORT (exact coordinates)
  Fallback → ip-api.com (approximate city-level, no API key needed)

Video source: IP camera MJPEG stream → http://192.0.0.2:8081

Requirements:
    pip install -r requirements_raspi.txt

Download shape predictor (run once):
    wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

Hardware wiring:
    LED   (+resistor 220Ω) → GPIO 17  (Physical Pin 11)
    Passive Buzzer (+)     → GPIO 27  (Physical Pin 13)
    NEO-6M GPS TX          → GPIO 15 / RXD (Physical Pin 10)  [optional]
    NEO-6M GPS RX          → GPIO 14 / TXD (Physical Pin 8)   [optional]
    All GND                → GND     (Physical Pin 6 or 9)

Gmail setup (do this once):
    1. Enable 2-Step Verification on your Google account
    2. Go to: myaccount.google.com → Security → App Passwords
    3. Create an App Password for "Mail"
    4. Paste the 16-character password into SENDER_APP_PASSWORD below
"""

import cv2
import dlib
import time
import threading
import smtplib
import getpass
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
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
BUZZER_DUTY      = 0.5      # Duty cycle 0.0–1.0  (0.5 = 50%)

EAR_CLOSED_RATIO  = 0.75    # EAR threshold = baseline_ear × this ratio
DROWSY_SECONDS    = 0.7     # Eyes must be closed this long before alert fires
ALERT_RESEND_SEC  = 5       # Heartbeat: re-assert GPIO every N seconds while drowsy
CALIBRATION_SECS  = 3       # Seconds to sample open-eye EAR baseline at startup
EMAIL_ALERT_AFTER = 6.0     # Seconds drowsy before email is sent (no response to buzzer)
EMAIL_COOLDOWN    = 120     # Minimum seconds between successive alert emails

# ── Email credentials ─────────────────────────
SENDER_EMAIL        = "anandhanachu785@gmail.com"      # Gmail that sends the alert
SENDER_APP_PASSWORD = "xxxx xxxx xxxx xxxx"           # Placeholder — entered at runtime
RECEIVER_EMAIL      = "anandhan_ec24@ug.cusat.in"     # Who receives the alert
DRIVER_NAME         = "Driver"                        # Name shown in email

# ── GPS module (NEO-6M) ───────────────────────
GPS_ENABLED = True                  # Set False if no GPS module connected
GPS_PORT    = "/dev/ttyAMA0"        # RPi UART port (/dev/ttyUSB0 for USB GPS dongle)
GPS_BAUD    = 9600                  # NEO-6M default baud rate


# ─────────────────────────────────────────────
# GPIO SETUP  (gpiozero)
# ─────────────────────────────────────────────
try:
    from gpiozero import LED as GpioLED, PWMOutputDevice

    led    = GpioLED(PIN_LED)
    buzzer = PWMOutputDevice(PIN_BUZZER, frequency=BUZZER_FREQ)
    buzzer.value = 0

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
        led.blink(on_time=1/LED_BLINK_HZ, off_time=1/LED_BLINK_HZ)
        buzzer.value = BUZZER_DUTY
    print("[GPIO] ALERT ON  — LED blinking, Buzzer sounding")


def gpio_alert_off():
    """Stop LED + silence passive buzzer."""
    if gpio_available:
        led.off()
        buzzer.value = 0
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
# GPS — NEO-6M module via serial (NMEA parsing)
# ─────────────────────────────────────────────
_gps_lat    = None      # latest latitude  (float degrees)
_gps_lon    = None      # latest longitude (float degrees)
_gps_lock   = threading.Lock()


def _parse_nmea_coord(value: str, direction: str) -> float:
    """
    Convert raw NMEA coordinate string to decimal degrees.
    e.g. '1142.1234', 'N'  →  11.702057
    """
    if not value:
        return None
    dot = value.index('.')
    degrees = float(value[:dot - 2])
    minutes = float(value[dot - 2:])
    decimal = degrees + minutes / 60.0
    if direction in ('S', 'W'):
        decimal = -decimal
    return decimal


def _gps_reader_thread():
    """
    Background thread: continuously reads NMEA sentences from the GPS module
    and updates _gps_lat / _gps_lon whenever a valid fix is received.
    """
    global _gps_lat, _gps_lon
    try:
        import serial
        with serial.Serial(GPS_PORT, GPS_BAUD, timeout=2) as gps_ser:
            print(f"[GPS] Connected on {GPS_PORT} at {GPS_BAUD} baud")
            while True:
                try:
                    line = gps_ser.readline().decode('ascii', errors='replace').strip()
                except Exception:
                    continue

                # GPRMC or GNRMC: Recommended Minimum Specific GPS Data
                if line.startswith(('$GPRMC', '$GNRMC')):
                    parts = line.split(',')
                    if len(parts) >= 7 and parts[2] == 'A':   # 'A' = active/valid fix
                        lat = _parse_nmea_coord(parts[3], parts[4])
                        lon = _parse_nmea_coord(parts[5], parts[6])
                        if lat is not None and lon is not None:
                            with _gps_lock:
                                _gps_lat = lat
                                _gps_lon = lon
    except ImportError:
        print("[GPS] pyserial not installed — GPS module disabled.")
    except Exception as e:
        print(f"[GPS] Error reading GPS module: {e}")


def _start_gps():
    """Start GPS reader in background thread if GPS is enabled."""
    if GPS_ENABLED:
        t = threading.Thread(target=_gps_reader_thread, daemon=True)
        t.start()
        print("[GPS] Background reader started. Waiting for satellite fix...")
    else:
        print("[GPS] GPS module disabled (GPS_ENABLED = False).")


def get_location() -> dict:
    """
    Get current location.
    Returns:
      - GPS coordinates (exact) if hardware GPS has a fix
      - IP-based location (approximate) as fallback
    Format: {'lat': float, 'lon': float, 'source': str, 'address': str}
    """
    # ── Try hardware GPS first ────────────────
    with _gps_lock:
        lat, lon = _gps_lat, _gps_lon

    if lat is not None and lon is not None:
        maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        return {
            "lat":     lat,
            "lon":     lon,
            "source":  "GPS (exact)",
            "address": f"{lat:.6f}, {lon:.6f}",
            "maps":    maps_url,
        }

    # ── Fallback: IP-based geolocation ───────
    try:
        resp = requests.get("http://ip-api.com/json", timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            lat = data["lat"]
            lon = data["lon"]
            city    = data.get("city",    "")
            region  = data.get("regionName", "")
            country = data.get("country", "")
            address = f"{city}, {region}, {country}"
            maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
            return {
                "lat":     lat,
                "lon":     lon,
                "source":  "IP geolocation (approximate)",
                "address": address,
                "maps":    maps_url,
            }
    except Exception as e:
        print(f"[Location] IP geolocation failed: {e}")

    # ── No location available ─────────────────
    return {
        "lat":     None,
        "lon":     None,
        "source":  "Unavailable",
        "address": "Location could not be determined",
        "maps":    None,
    }


# ─────────────────────────────────────────────
# EMAIL ALERT  (Gmail SMTP — built-in smtplib)
# ─────────────────────────────────────────────
_last_email_ts = 0.0
_email_lock    = threading.Lock()


def send_email(subject: str, body: str, maps_url: str = None):
    """Send email via Gmail SMTP in a background thread."""
    def _send():
        try:
            msg            = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = RECEIVER_EMAIL

            text_part = MIMEText(body, "plain")

            # Build HTML — highlight alert in red, safe in green
            is_alert  = "ALERT" in subject
            hdr_color = "#cc0000" if is_alert else "#007700"
            maps_html = (
                f'<p><a href="{maps_url}" style="background:#1a73e8;color:#fff;'
                f'padding:10px 20px;border-radius:6px;text-decoration:none;'
                f'font-weight:bold;">📍 Open in Google Maps</a></p>'
                if maps_url else ""
            )
            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;padding:20px;">
              <h2 style="color:{hdr_color};">{subject}</h2>
              {maps_html}
              <pre style="background:#f5f5f5;padding:15px;border-radius:8px;
                          font-size:14px;line-height:1.7;">{body}</pre>
              <p style="color:#888;font-size:12px;margin-top:20px;">
                Sent automatically by Drowsiness Detection System (Raspberry Pi)
              </p>
            </body></html>
            """
            html_part = MIMEText(html_body, "html")
            msg.attach(text_part)
            msg.attach(html_part)

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

            print(f"[Email] Sent → {RECEIVER_EMAIL} | {subject}")

        except smtplib.SMTPAuthenticationError:
            print("[Email] ERROR: Authentication failed. Check SENDER_EMAIL / SENDER_APP_PASSWORD.")
        except Exception as e:
            print(f"[Email] ERROR: {e}")

    threading.Thread(target=_send, daemon=True).start()


def send_drowsy_alert(ear: float, elapsed: float):
    """Send emergency alert email with live location."""
    global _last_email_ts
    now = time.time()

    with _email_lock:
        if now - _last_email_ts < EMAIL_COOLDOWN:
            return
        _last_email_ts = now

    loc       = get_location()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject   = f"🚨 DROWSY ALERT — {DRIVER_NAME} unresponsive for {elapsed:.0f}s"

    loc_line  = (
        f"GPS Location : {loc['address']}\n"
        f"Coordinates  : {loc['lat']:.6f}, {loc['lon']:.6f}\n"
        f"Location src : {loc['source']}\n"
        f"Maps link    : {loc['maps'] or 'N/A'}"
    ) if loc["lat"] else (
        f"Location     : {loc['address']}"
    )

    body = (
        f"⚠️  DROWSINESS ALERT — IMMEDIATE ATTENTION REQUIRED\n"
        f"{'='*48}\n"
        f"Driver       : {DRIVER_NAME}\n"
        f"Time         : {timestamp}\n"
        f"Unresponsive : {elapsed:.1f} seconds\n"
        f"EAR value    : {ear:.3f}  (threshold: {EAR_THRESHOLD:.3f})\n"
        f"{'─'*48}\n"
        f"{loc_line}\n"
        f"{'='*48}\n\n"
        f"The driver has NOT responded to the buzzer alarm.\n"
        f"Please check on the driver immediately.\n"
    )
    send_email(subject, body, maps_url=loc["maps"])
    print(f"[Email] DROWSY ALERT sent — {elapsed:.1f}s unresponsive | Location: {loc['source']}")


def send_safe_alert(ear: float, total_drowsy: float):
    """Send follow-up 'driver is now awake' email with location."""
    loc       = get_location()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject   = f"✅ DRIVER SAFE — {DRIVER_NAME} is now awake"

    loc_line  = (
        f"GPS Location : {loc['address']}\n"
        f"Coordinates  : {loc['lat']:.6f}, {loc['lon']:.6f}\n"
        f"Maps link    : {loc['maps'] or 'N/A'}"
    ) if loc["lat"] else (
        f"Location     : {loc['address']}"
    )

    body = (
        f"✅  SITUATION RESOLVED — DRIVER AWAKE\n"
        f"{'='*48}\n"
        f"Driver       : {DRIVER_NAME}\n"
        f"Time         : {timestamp}\n"
        f"EAR now      : {ear:.3f}  (threshold: {EAR_THRESHOLD:.3f})\n"
        f"Drowsy for   : {total_drowsy:.1f} seconds total\n"
        f"{'─'*48}\n"
        f"{loc_line}\n"
        f"{'='*48}\n\n"
        f"The driver appears awake and alert now.\n"
    )
    send_email(subject, body, maps_url=loc["maps"])
    print(f"[Email] SAFE alert sent — awake after {total_drowsy:.1f}s | Location: {loc['source']}")


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
# GPS START
# ─────────────────────────────────────────────
_start_gps()


# ─────────────────────────────────────────────
# IP CAMERA SETUP
# ─────────────────────────────────────────────
print(f"[Camera] Connecting to: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
# CALIBRATION
# ─────────────────────────────────────────────
print(f"[Calibration] Keep your eyes OPEN for {CALIBRATION_SECS} seconds...")
ear_samples   = []
calib_start   = time.time()
EAR_THRESHOLD = 0.22

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
        break

if ear_samples:
    baseline_ear  = np.mean(ear_samples)
    EAR_THRESHOLD = baseline_ear * EAR_CLOSED_RATIO
    print(f"[Calibration] Baseline EAR: {baseline_ear:.3f}  →  Threshold: {EAR_THRESHOLD:.3f}")
else:
    print(f"[Calibration] No face detected. Using default threshold: {EAR_THRESHOLD:.3f}")


# ─────────────────────────────────────────────
# APP PASSWORD — entered securely at runtime
# ─────────────────────────────────────────────
SENDER_APP_PASSWORD = getpass.getpass(
    f"[Email] Enter Gmail App Password for {SENDER_EMAIL}: "
)
print("[Email] App Password received.")

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
sleep_state       = False
eyes_closed_since = None
last_alert_ts     = 0.0
email_sent        = False

print(
    f"\n[System] Running.\n"
    f"  LED      : GPIO {PIN_LED}\n"
    f"  Buzzer   : GPIO {PIN_BUZZER} (PWM)\n"
    f"  Email    : alert after {EMAIL_ALERT_AFTER}s → {RECEIVER_EMAIL}\n"
    f"  Location : GPS {'enabled' if GPS_ENABLED else 'disabled'} | IP fallback active\n"
    f"  Ctrl+C   : quit\n"
)


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
                email_sent  = False
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

            # Stage 1 (0.7s): GPIO alert — LED + Buzzer
            if elapsed >= DROWSY_SECONDS:
                if not sleep_state:
                    gpio_alert_on()
                    last_alert_ts = now
                    sleep_state   = True
                    print(f"[Alert] DROWSY — EAR={ear:.3f} < {EAR_THRESHOLD:.3f}")
                elif now - last_alert_ts >= ALERT_RESEND_SEC:
                    gpio_alert_on()
                    last_alert_ts = now
                    print(f"[Alert] DROWSY heartbeat — EAR={ear:.3f}, elapsed={elapsed:.1f}s")

            # Stage 2 (6s): Email alert with live location
            if elapsed >= EMAIL_ALERT_AFTER and not email_sent:
                send_drowsy_alert(ear, elapsed)
                email_sent = True

        else:
            # ── DRIVER AWAKE ─────────────────────────
            if sleep_state:
                total_drowsy = (now - eyes_closed_since) if eyes_closed_since else 0
                gpio_alert_off()
                if email_sent:
                    send_safe_alert(ear, total_drowsy)
                sleep_state       = False
                email_sent        = False
                eyes_closed_since = None
                print(f"[Alert] AWAKE  — EAR={ear:.3f}")
            else:
                eyes_closed_since = None

except KeyboardInterrupt:
    print("\n[System] Stopped by user.")

finally:
    if sleep_state:
        gpio_alert_off()
    gpio_cleanup()
    cap.release()
    print("[System] Done.")
