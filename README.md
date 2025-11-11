# 🎶 AirBeats – Virtual Music Using Hand Gestures

**AirBeats** is a real-time **webcam-based virtual music app** that lets you play different instruments in the air using just your hands.  
It uses **MediaPipe** for hand tracking, **OpenCV** for live camera feed, and **Pygame** for sound playback.

---

## 🧠 Project Overview

When you run AirBeats, your webcam turns into a virtual music board:
- 🎥 Your live camera feed appears on screen.
- 🖐 Four instrument icons (Drum, Tabla, Guitar, Piano) appear at the corners.
- ✋ When your hand moves into a zone, the corresponding sound plays instantly.
- 💫 Icons are semi-transparent so you can still see your fingers clearly.
- 🎹 Press **`Q`** anytime to quit the app.

---

## 🧰 Tech Stack

| Component | Library / Tool |
|------------|----------------|
| Computer Vision | [OpenCV](https://opencv.org/) |
| Hand Tracking | [MediaPipe](https://developers.google.com/mediapipe) |
| Audio Playback | [Pygame](https://www.pygame.org/) |
| Language | Python 3.9+ |
| Platform | Works on Windows / macOS / Linux |

---

## 🗂️ Folder Structure

air_instruments/
├── main.py
└── assets/
├── sounds/
│ ├── drum.wav
│ ├── tabla.wav
│ ├── guitar.wav
│ └── piano.wav
└── icons/
├── drum.png
├── tabla.png
├── guitar.png
└── piano.png


---

## ⚙️ Installation & Setup

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/<your-username>/AirBeats.git
   cd AirBeats

2. **Create a Virtual Environment**

python -m venv venv
venv\Scripts\activate     # (on Windows)
# OR
source venv/bin/activate  # (on macOS/Linux)

3. **Install Required Libraries**

pip install opencv-python mediapipe pygame numpy

4. **Run the App**

python main.py

---

🪄 Features

✅ Real-time hand tracking using your webcam
✅ Play 4 instruments by moving your hand into different corners
✅ Semi-transparent icons that don’t block the feed
✅ Smooth sound playback with cooldown logic
✅ Press Q anytime to exit safely

---

💡 Future Enhancements

Add more instruments dynamically

Gesture-based play (like pinch, wave, tap)

Glow animation when an instrument is played

Record and save custom AirBeats performances

Web-based version using TensorFlow.js


👨‍💻 Author

Anshool Dahale
CSE (AI & ML) @ PES University
📧 anshooldahale08@gmail.com




