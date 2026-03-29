# Ishara_Coffee

> *Ishara* (इशारा) — Hindi/Bengali for **gesture** or **signal**. Because your hands do the talking. ☕

A real-time hand gesture recognition system built with **MediaPipe** and **Python** — detects single-hand gestures and two-hand combos, triggers popup images with audio, and streams live data to a web dashboard.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-SocketIO-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Category | Gestures |
|---|---|
| 🤚 Single-hand | `thumbs_up`, `peace`, `fist`, `open_hand`, `pointing`, `rock_on`, `call_me` |
| 🙌 Two-hand combos | `boom_boom`, `okay`, `total_peace`, `absolute_cinema` |
| 🎨 Custom gestures | Record any gesture live via the app |
| 🖼️ Popup display | Triggered image shown via Pygame on gesture detection |
| 🌐 Web dashboard | Real-time gesture stream viewable in browser |
| 🔊 Audio playback | Audio clip plays in sync with image popup |

---

## 📁 Project Structure

```
Ishara_Coffee/
├── main.py                  # Main application
├── code.html                # Web dashboard
├── hand_landmarker.task     # MediaPipe model (download separately)
├── custom_gestures.json     # Auto-generated when you record gestures
└── assets/                  # ⚠️ NOT included — add your own (see below)
    ├── images/
    └── audio/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Ishara_Coffee.git
cd Ishara_Coffee
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy pygame Pillow flask flask-socketio
```

> **Python 3.8+** is recommended.

### 3. Download the MediaPipe Model

Download `hand_landmarker.task` from the official MediaPipe release and place it in the project root:

🔗 [Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

```
Ishara_Coffee/
└── hand_landmarker.task   ✅ place it here
```

---

## 🖼️ Setting Up Assets

> **Assets (images and audio) are NOT included in this repo.** You need to bring your own — that's the whole point. Make it personal. ✌️

### Folder Structure

Create the following structure inside the project root:

```
assets/
├── images/
│   ├── thumbs_up.jpg
│   ├── peace.jpg
│   ├── fist.jpg
│   ├── open_hand.jpg
│   ├── pointing.jpg
│   ├── rock_on.jpg
│   ├── call_me.jpg
│   ├── boom_boom.jpg
│   ├── okay.jpg
│   ├── total_peace.jpg
│   └── absolute_cinema.jpg
└── audio/
    ├── thumbs_up.mp3
    ├── peace.mp3
    ├── fist.mp3
    ├── open_hand.mp3
    ├── pointing.mp3
    ├── rock_on.mp3
    ├── call_me.mp3
    ├── boom_boom.mp3
    ├── okay.mp3
    ├── total_peace.mp3
    └── absolute_cinema.mp3
```

### Where to get assets

**Images:**
- Use any `.jpg` image of your choice for each gesture (memes, movie stills, custom artwork — anything)
- Recommended size: `400x400px` or larger for best popup quality
- Free sources: [Unsplash](https://unsplash.com), [Pexels](https://pexels.com), or create your own

**Audio:**
- Use any `.mp3` clip (sound effects, music snippets, voice clips, etc.)
- Keep clips short — **2 to 5 seconds** works best
- Free sources: [Pixabay](https://pixabay.com/sound-effects/), [Freesound](https://freesound.org)

> ⚠️ **File names must match exactly** as listed above (case-sensitive). The app will not trigger the popup if the file is missing or misnamed.

---

## 🚀 Usage

```bash
python main.py
```

Your webcam will activate and Ishara_Coffee will begin detecting gestures in real time.

### ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `R` | Record a new custom gesture |
| `D` | Delete a saved custom gesture |
| `L` | List all saved custom gestures |
| `Q` | Quit the application |

### 🌐 Web Dashboard

Open `code.html` directly in your browser, or if the Flask server is running, visit:

```
http://localhost:5000
```

---

## 🎨 Recording Custom Gestures

You can add any gesture beyond the built-in ones:

1. Run `main.py`
2. Press `R`
3. Enter a name for your gesture (e.g., `thumbs_down`)
4. Hold the gesture steady in front of the camera for **~3 seconds**
5. It gets saved automatically to `custom_gestures.json`

> Add a matching image and audio file in `assets/images/` and `assets/audio/` using the same name to enable popup + audio for your custom gesture.

---

## 🛠️ Tech Stack

- [MediaPipe](https://developers.google.com/mediapipe) — Hand landmark detection
- [OpenCV](https://opencv.org/) — Webcam feed & frame processing
- [Pygame](https://www.pygame.org/) — Image popup & audio playback
- [Flask + SocketIO](https://flask-socketio.readthedocs.io/) — Real-time web dashboard
- [NumPy](https://numpy.org/) + [Pillow](https://python-pillow.org/) — Data handling & image processing

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> Made with ❤️ by [Ronit Shaw](https://github.com/beingRonit)
