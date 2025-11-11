import cv2
import mediapipe as mp
import pygame
import os
import time
import numpy as np

# ----------------- Config -----------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_ICONS = os.path.join(BASE_PATH, "assets", "icons")
ASSETS_SOUNDS = os.path.join(BASE_PATH, "assets", "sounds")

# Frame (increase size)
FRAME_W = 1280
FRAME_H = 720

# box size for corner icons
BOX_SIZE = 200   # icon square size in px
COOLDOWN_FRAMES = 12  # frames to wait before re-trigger (approx)

# instrument mapping (keys should match filenames)
INSTRUMENTS = ["drum", "tabla", "guitar", "piano"]

# ----------------- Init audio -----------------
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()

sounds = {}
for ins in INSTRUMENTS:
    sound_path = os.path.join(ASSETS_SOUNDS, f"{ins}.wav")
    if os.path.isfile(sound_path):
        sounds[ins] = pygame.mixer.Sound(sound_path)
    else:
        print(f"[WARN] Sound not found: {sound_path}. This instrument will be silent.")
        sounds[ins] = None

# ----------------- Load icons -----------------
icons = {}
for ins in INSTRUMENTS:
    icon_path = os.path.join(ASSETS_ICONS, f"{ins}.png")
    img = None
    if os.path.isfile(icon_path):
        img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Failed to read icon (maybe unsupported format): {icon_path}")
    else:
        print(f"[WARN] Icon not found: {icon_path}")
    icons[ins] = img  # may be None

# ----------------- Overlay function (safe, handles no-alpha) -----------------
def overlay_with_alpha(background, overlay, x, y, size=(BOX_SIZE, BOX_SIZE), opacity=0.6):
    """Overlay `overlay` image (may have alpha) onto background at x,y resized to size.
       opacity multiplies the overlay alpha (0..1). Returns background (modified)."""
    if overlay is None:
        return background

    # Resize overlay to desired size
    overlay_resized = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay_resized.shape[:2]

    # prepare alpha mask
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3].astype(float) / 255.0
        overlay_rgb = overlay_resized[:, :, :3].astype(float)
    else:
        # no alpha channel -> fully opaque
        alpha = np.ones((h, w), dtype=float)
        overlay_rgb = overlay_resized.astype(float)

    alpha = np.clip(alpha * opacity, 0.0, 1.0)
    alpha = alpha[..., None]  # shape (h,w,1)

    # clipping ROI if out of bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)

    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background  # nothing to draw

    roi = background[y1:y2, x1:x2].astype(float)
    overlay_part = overlay_rgb[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha_part = alpha[ov_y1:ov_y2, ov_x1:ov_x2]

    blended = (alpha_part * overlay_part + (1 - alpha_part) * roi).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background

# ----------------- MediaPipe -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# ----------------- Zones (corners) -----------------
def get_corner_zones(frame_w, frame_h, box_size=BOX_SIZE, margin=20):
    return {
        "drum":   (margin, margin, margin + box_size, margin + box_size),                      # top-left
        "tabla":  (frame_w - margin - box_size, margin, frame_w - margin, margin + box_size),  # top-right
        "guitar": (margin, frame_h - margin - box_size, margin + box_size, frame_h - margin),  # bottom-left
        "piano":  (frame_w - margin - box_size, frame_h - margin - box_size, frame_w - margin, frame_h - margin)  # bottom-right
    }

# ----------------- Webcam -----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
time.sleep(0.2)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions / connection.")

cooldown = {ins: 0 for ins in INSTRUMENTS}

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    fh, fw = frame.shape[:2]
    zones = get_corner_zones(fw, fh)

    # draw icons (semi-transparent)
    for ins, (x1, y1, x2, y2) in zones.items():
        icon = icons.get(ins)
        if icon is not None:
            overlay_with_alpha(frame, icon, x1, y1, size=(x2 - x1, y2 - y1), opacity=0.55)
        else:
            # fallback rectangle with label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.putText(frame, ins.upper(), (x1 + 8, y1 + (y2 - y1)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    # hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark[8]  # index tip
            fx, fy = int(lm.x * fw), int(lm.y * fh)
            fingertip_positions.append((fx, fy))
            cv2.circle(frame, (fx, fy), 8, (0, 255, 255), -1)

    # collision detection & play sounds
    for fx, fy in fingertip_positions:
        for ins, (x1, y1, x2, y2) in zones.items():
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                # highlight border
                cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (0, 220, 0), 3)
                if cooldown[ins] == 0:
                    snd = sounds.get(ins)
                    if snd:
                        try:
                            snd.play()
                        except Exception as e:
                            print("Error playing sound:", e)
                    cooldown[ins] = COOLDOWN_FRAMES

    # update cooldowns
    for k in cooldown:
        if cooldown[k] > 0:
            cooldown[k] -= 1

    # UI text
    cv2.putText(frame, "Virtual Instruments - Press 'q' to quit", (20, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2)

    cv2.imshow("Virtual Instruments", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
