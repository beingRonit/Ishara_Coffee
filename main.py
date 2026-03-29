"""
╔══════════════════════════════════════════════════════════════╗
║       ISHARA_COFFEE — GESTURE RECOGNITION SYSTEM            ║
║  Features:                                                   ║
║    • MediaPipe Tasks hand landmark detection                 ║
║    • Rule-based gesture classifier (7 gestures)             ║
║    • ML model inference (RandomForest, if trained)          ║
║    • KNN custom gesture recognizer (live + web recording)   ║
║    • Two-hand combo detection (4 combos)                     ║
║    • Pygame popup image display                             ║
║    • Audio playback per gesture (via web dashboard only)    ║
║    • Stability buffer + cooldown anti-spam                   ║
║    • Full logging via Python logging module                 ║
║    • Real FPS/Latency stats emitted via SocketIO            ║
║    • /retrain endpoint for on-demand ML retraining          ║
║    • /gestures CRUD API for web-based custom gestures       ║
╚══════════════════════════════════════════════════════════════╝

Controls (while camera window is active):
  R  =  Record a new custom gesture
  D  =  Delete a custom gesture
  L  =  List all saved custom gestures
  Q  =  Quit

Requirements:
  pip install opencv-python mediapipe numpy pygame Pillow joblib scikit-learn flask flask-socketio simple-websocket

Assets expected in:
  assets/images/<gesture_name>.jpg
  assets/audio/<gesture_name>.mp3
  hand_landmarker.task        (MediaPipe model file)
  gesture_model.pkl           (optional — trained via train_model.py)
  label_encoder.pkl           (optional — trained via train_model.py)
  custom_gestures.json        (auto-created when you record)
"""

# type: ignore
# pylint: disable=no-member
import cv2        # type: ignore
import mediapipe as mp  # type: ignore
import pygame     # type: ignore
import os
import json
import time
import threading
import logging
import numpy as np  # type: ignore
from typing import Any, Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ══════════════════════════════════════════════════════════════
#  LOGGING SETUP — replaces all print() calls
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ishara.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ISHARA")

# ── Flask SocketIO for Web Dashboard ──
try:
    from flask import Flask, send_file, send_from_directory, Response, jsonify, request
    from flask_socketio import SocketIO
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    log.warning("Flask not found. Web dashboard disabled.")

flask_app = None
socketio  = None

_latest_frame = None
_frame_lock   = threading.Lock()

_stats_lock = threading.Lock()
_real_stats = {"fps": 0.0, "latency_ms": 0, "hand_count": 0, "mode": "DETECT"}

_custom_db_ref:  Any = None
_ml_model_ref:   Any = None
_retrain_lock = threading.Lock()
_record_request: dict = {}

if FLASK_AVAILABLE:
    flask_app = Flask(__name__)
    socketio  = SocketIO(flask_app, cors_allowed_origins="*", async_mode='threading')

    @flask_app.route('/')
    def index():
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.html")
        return send_file(html_path)

    @flask_app.route('/assets/<path:filename>')
    def serve_assets(filename):
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        return send_from_directory(assets_dir, filename)

    @flask_app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                with _frame_lock:
                    if _latest_frame is None:
                        time.sleep(0.05)
                        continue
                    frame_copy = _latest_frame.copy()
                ok, jpeg = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           jpeg.tobytes() + b'\r\n')
                time.sleep(0.033)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @flask_app.route('/stats')
    def stats_route():
        with _stats_lock:
            return jsonify(_real_stats)

    @flask_app.route('/camera_check')
    def camera_check():
        cap = cv2.VideoCapture(0)
        ok  = cap.isOpened()
        cap.release()
        return jsonify({"ok": ok})

    @flask_app.route('/gestures', methods=['GET'])
    def list_gestures_route():
        if _custom_db_ref is None:
            return jsonify({"error": "not ready"}), 503
        return jsonify(_custom_db_ref.list_gestures())

    @flask_app.route('/gestures/<name>', methods=['DELETE'])
    def delete_gesture_route(name):
        if _custom_db_ref is None:
            return jsonify({"error": "not ready"}), 503
        ok = _custom_db_ref.delete(name)
        if ok:
            log.info("Web: deleted custom gesture '%s'", name)
            return jsonify({"deleted": name})
        return jsonify({"error": "not found"}), 404

    @flask_app.route('/gestures/record', methods=['POST'])
    def start_record_route():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip().lower().replace(" ", "_")
        if not name:
            return jsonify({"error": "name required"}), 400
        _record_request["name"]  = name
        _record_request["ready"] = True
        log.info("Web: recording request queued for '%s'", name)
        return jsonify({"status": "recording_queued", "name": name})

    @flask_app.route('/retrain', methods=['POST'])
    def retrain_route():
        if not JOBLIB_AVAILABLE:
            return jsonify({"error": "joblib not installed"}), 500
        if _custom_db_ref is None:
            return jsonify({"error": "not ready"}), 503

        def do_retrain():
            with _retrain_lock:
                log.info("RETRAIN: Starting on-demand retraining...")
                try:
                    from sklearn.ensemble import RandomForestClassifier  # type: ignore
                    from sklearn.preprocessing import LabelEncoder      # type: ignore
                    import joblib as jl  # type: ignore

                    db = _custom_db_ref.db
                    if not db:
                        log.warning("RETRAIN: No data to train on.")
                        socketio.emit("retrain_result", {"status": "error", "msg": "No data"})
                        return

                    X, y = [], []
                    for label, samples in db.items():
                        for s in samples:
                            X.append(s)
                            y.append(label)

                    le    = LabelEncoder()
                    y_enc = le.fit_transform(y)
                    clf   = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X, y_enc)

                    jl.dump(clf, "gesture_model.pkl")
                    jl.dump(le,  "label_encoder.pkl")

                    if _ml_model_ref is not None:
                        _ml_model_ref.model     = clf
                        _ml_model_ref.encoder   = le
                        _ml_model_ref.available = True

                    log.info("RETRAIN: Done. Classes: %s", list(le.classes_))
                    socketio.emit("retrain_result", {"status": "ok", "classes": list(le.classes_)})
                except Exception as exc:
                    log.error("RETRAIN: Failed: %s", exc)
                    socketio.emit("retrain_result", {"status": "error", "msg": str(exc)})

        threading.Thread(target=do_retrain, daemon=True).start()
        return jsonify({"status": "retraining_started"})

    def run_socketio():
        socketio.run(flask_app, host="127.0.0.1", port=5001, debug=False, use_reloader=False)

    threading.Thread(target=run_socketio, daemon=True).start()
    log.info("SocketIO server started. Open: http://127.0.0.1:5001")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    log.warning("joblib not found. ML model disabled.")


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    CAMERA_INDEX        = 0
    TARGET_FPS          = 30
    STABILITY_FRAMES    = 10
    COOLDOWN_SECONDS    = 2.0
    KNN_K               = 5
    KNN_THRESHOLD       = 0.60
    ML_THRESHOLD        = 0.85
    FINGER_UP_MARGIN    = 0.02
    POPUP_SIZE          = (500, 500)
    DISPLAY_FRAMES      = 90
    SAMPLES_PER_GESTURE = 60
    CUSTOM_DB_FILE      = "custom_gestures.json"

    BUILTIN_ASSETS = {
        "thumbs_up":  {"image": "assets/images/Thumbs up.jpg", "audio": "assets/audio/truth_thumb.mp3",      "label": "THUMBS UP"},
        "peace":      {"image": "assets/images/peace.jpg",     "audio": "assets/audio/disturbing_peace.mp3", "label": "PEACE"},
        "fist":       {"image": "assets/images/fist.jpg",      "audio": "assets/audio/one-punch.mp3",        "label": "FIST"},
        "open_hand":  {"image": "assets/images/open_hand.jpg", "audio": "assets/audio/open_f.mp3",           "label": "OPEN HAND"},
        "pointing":   {"image": "assets/images/pointing.jpg",  "audio": "assets/audio/nerd_pointing.mp3",    "label": "POINTING"},
        "rock_on":    {"image": "assets/images/rock_on.jpg",   "audio": "assets/audio/rock_on.mp3",          "label": "ROCK ON"},
        "call_me":    {"image": "assets/images/call_me.jpg",   "audio": "assets/audio/call_me.mp3",          "label": "CALL ME"},
    }

    COMBO_MAP = {
        "fist+fist":           "boom_boom",
        "thumbs_up+thumbs_up": "okay",
        "peace+peace":         "total_peace",
        "open_hand+open_hand": "absolute_cinema",
    }

    COMBO_ASSETS = {
        "boom_boom":       {"image": "assets/images/boom_boom.jpg",       "audio": "assets/audio/boom_boom.mp3",       "label": "BOOM BOOM"},
        "okay":            {"image": "assets/images/okay.jpg",            "audio": "assets/audio/okay.mp3",            "label": "OKAY"},
        "total_peace":     {"image": "assets/images/total_peace.jpg",     "audio": "assets/audio/total_peace.mp3",     "label": "TOTAL PEACE"},
        "absolute_cinema": {"image": "assets/images/absolute_cinema.jpg", "audio": "assets/audio/absolute_cinema.mp3", "label": "ABSOLUTE CINEMA"},
    }


# ══════════════════════════════════════════════════════════════
#  ASSET MANAGER
# ══════════════════════════════════════════════════════════════
class AssetManager:
    def __init__(self, builtin_map: dict, combo_map: dict, popup_size: tuple):
        self.popup_size      = popup_size
        self.cv2_images: dict  = {}
        self.pygame_surfs: dict = {}
        self.audio_paths: dict  = {}
        self.labels: dict       = {}
        log.info("Loading built-in assets...")
        self._load_map(builtin_map)
        log.info("Loading combo assets...")
        self._load_map(combo_map)
        log.info("Assets loaded.")

    def _load_map(self, asset_map: dict):
        for name, data in asset_map.items():
            img_path   = data.get("image", "")
            audio_path = data.get("audio", "")
            label      = data.get("label", name.upper())

            cv2_img = None
            if os.path.exists(img_path):
                raw = cv2.imread(img_path)
                if raw is not None:
                    cv2_img = cv2.resize(raw, (320, 320))
                else:
                    log.warning("cv2 could not decode image for '%s'", name)
            else:
                log.warning("Image not found for '%s': %s", name, img_path)

            pg_surf = None
            if os.path.exists(img_path):
                try:
                    surf    = pygame.image.load(img_path)
                    pg_surf = pygame.transform.scale(surf, self.popup_size)
                except Exception as exc:
                    log.warning("Pygame image error for '%s': %s", name, exc)

            audio = audio_path if os.path.exists(audio_path) else None
            if audio is None:
                log.warning("Audio not found for '%s': %s", name, audio_path)

            self.cv2_images[name]   = cv2_img
            self.pygame_surfs[name] = pg_surf
            self.audio_paths[name]  = audio
            self.labels[name]       = label

    def get_pygame_surf(self, name: str) -> Any:   return self.pygame_surfs.get(name)
    def get_audio(self, name: str) -> Optional[str]: return self.audio_paths.get(name)
    def get_label(self, name: str) -> str:          return self.labels.get(name, name.upper())


# ══════════════════════════════════════════════════════════════
#  CUSTOM GESTURE DATABASE (KNN)
# ══════════════════════════════════════════════════════════════
class CustomGestureDB:
    def __init__(self, filepath: str, k: int, threshold: float):
        self.filepath  = filepath
        self.k         = k
        self.threshold = threshold
        self.db        = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                data = json.load(f)
            log.info("KNN: Loaded %d custom gesture(s): %s", len(data), list(data.keys()))
            return data
        return {}

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.db, f, indent=2)
        log.info("KNN: Saved to %s", self.filepath)

    def add_samples(self, name: str, samples: list):
        if name not in self.db:
            self.db[name] = []
        self.db[name].extend(samples)
        self.save()
        log.info("KNN: '%s' now has %d samples", name, len(self.db[name]))

    def delete(self, name: str) -> bool:
        if name in self.db:
            del self.db[name]
            self.save()
            return True
        return False

    def list_gestures(self):
        return {name: len(samples) for name, samples in self.db.items()}

    @staticmethod
    def extract_features(landmarks) -> list:
        raw = []
        for lm in landmarks:
            raw.extend([lm.x, lm.y, lm.z])
        wrist = raw[:3]
        normalised = []
        for i in range(0, len(raw), 3):
            normalised.extend([raw[i]-wrist[0], raw[i+1]-wrist[1], raw[i+2]-wrist[2]])
        return normalised

    def predict(self, landmarks) -> tuple:
        if not self.db:
            return "unknown", 0.0
        query     = np.array(self.extract_features(landmarks))
        distances = []
        for name, samples in self.db.items():
            for sample in samples:
                distances.append((np.linalg.norm(query - np.array(sample)), name))
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        votes: dict = {}
        for _, name in top_k:
            votes[name] = votes.get(name, 0) + 1
        if not votes:
            return "unknown", 0.0
        winner     = max(votes.items(), key=lambda x: x[1])[0]
        confidence = votes[winner] / self.k
        if confidence < self.threshold:
            return "unknown", confidence
        return winner, confidence


# ══════════════════════════════════════════════════════════════
#  ML MODEL WRAPPER
# ══════════════════════════════════════════════════════════════
class MLModel:
    def __init__(self, model_path: str, encoder_path: str, threshold: float):
        self.model     = None
        self.encoder   = None
        self.threshold = threshold
        self.available = False
        if not JOBLIB_AVAILABLE:
            return
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.model     = joblib.load(model_path)
                self.encoder   = joblib.load(encoder_path)
                self.available = True
                log.info("ML: Model loaded. Classes: %s", list(self.encoder.classes_))
            except Exception as exc:
                log.error("ML: Failed to load model: %s", exc)
        else:
            log.info("ML: No trained model found. Use /retrain endpoint to enable.")

    def predict(self, landmarks) -> tuple:
        if not self.available:
            return "unknown", 0.0
        features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
        try:
            idx        = self.model.predict(features)[0]
            confidence = float(self.model.predict_proba(features).max())
            gesture    = self.encoder.inverse_transform([idx])[0]
            if confidence < self.threshold:
                return "unknown", confidence
            return gesture, confidence
        except Exception as exc:
            log.error("ML: Prediction error: %s", exc)
            return "unknown", 0.0


# ══════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════
class GestureClassifier:
    def __init__(self, ml_model: MLModel, custom_db: CustomGestureDB, finger_margin: float = 0.02):
        self.ml     = ml_model
        self.knn    = custom_db
        self.margin = finger_margin

    @staticmethod
    def _finger_states(landmarks, is_left: bool, margin: float) -> list:
        tip = landmarks[4]; mcp = landmarks[2]
        thumb_up = (tip.x > mcp.x) if is_left else (tip.x < mcp.x)
        fingers = [thumb_up]
        for t_idx, p_idx in zip([8,12,16,20], [6,10,14,18]):
            fingers.append(landmarks[t_idx].y < landmarks[p_idx].y - margin)
        return fingers

    @staticmethod
    def _rule_classify(fingers: list) -> str:
        thumb, index, middle, ring, pinky = fingers
        if thumb and not index and not middle and not ring and not pinky: return "thumbs_up"
        if not thumb and index and middle and not ring and not pinky:     return "peace"
        if not any(fingers):                                               return "fist"
        if all(fingers):                                                   return "open_hand"
        if not thumb and index and not middle and not ring and not pinky: return "pointing"
        if thumb and index and not middle and not ring and pinky:          return "rock_on"
        if thumb and not index and not middle and not ring and pinky:      return "call_me"
        return "unknown"

    def classify(self, landmarks, handedness_label: str) -> tuple:
        is_left = (handedness_label == "Right")
        if self.ml.available:
            gesture, conf = self.ml.predict(landmarks)
            if gesture != "unknown": return gesture, conf, "ml"
        gesture, conf = self.knn.predict(landmarks)
        if gesture != "unknown": return gesture, conf, "knn"
        fingers = self._finger_states(landmarks, is_left, self.margin)
        gesture = self._rule_classify(fingers)
        conf    = 0.9 if gesture != "unknown" else 0.0
        return gesture, conf, "rule"


# ══════════════════════════════════════════════════════════════
#  STABILITY BUFFER
# ══════════════════════════════════════════════════════════════
class StabilityBuffer:
    def __init__(self, n: int):
        self.n = n; self.buffer: list = []

    def update(self, value: str) -> str:
        self.buffer.append(value)
        if len(self.buffer) > self.n: self.buffer.pop(0)
        if len(self.buffer) == self.n and len(set(self.buffer)) == 1: return self.buffer[0]
        return "unknown"

    def reset(self) -> None: self.buffer.clear()


# ══════════════════════════════════════════════════════════════
#  COOLDOWN GATE
# ══════════════════════════════════════════════════════════════
class CooldownGate:
    def __init__(self, seconds: float):
        self.seconds = seconds; self.last_name: Optional[str] = None; self.last_time = 0.0

    def should_trigger(self, name: str) -> bool:
        if not name or name in ("unknown", "none"): return False
        now = time.time()
        if name != self.last_name or now - self.last_time > self.seconds:
            self.last_name = name; self.last_time = now; return True
        return False

    def reset(self) -> None: self.last_name = None; self.last_time = 0.0


# ══════════════════════════════════════════════════════════════
#  PYGAME POPUP DISPLAY
# ══════════════════════════════════════════════════════════════
class PopupDisplay:
    def __init__(self, size: tuple):
        self.screen: Any   = pygame.display.set_mode(size)
        pygame.display.set_caption("Gesture Display")
        self.size          = size
        self.font_big      = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_idle     = pygame.font.SysFont("Arial", 26)
        self.surface: Any  = None
        self.timer         = 0
        self.label         = ""
        self.is_combo      = False
        self.audio_playing = False

    def trigger(self, surface: Any, label: str, duration: int,
                is_combo: bool = False, audio_path: str = None) -> None:
        self.surface = surface; self.timer = duration
        self.label   = label;   self.is_combo = is_combo; self.audio_playing = False
        if audio_path and os.path.exists(audio_path):
            try:
                pygame.mixer.music.load(audio_path)
                self.timer = int(pygame.mixer.Sound(audio_path).get_length() * 60)
                pygame.mixer.music.play()
                self.audio_playing = True
            except Exception as exc:
                log.error("Audio error %s: %s", audio_path, exc)

    def tick(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
        if self.timer > 0 and self.surface is not None:
            self.screen.blit(self.surface, (0, 0))
            if self.is_combo:
                badge = self.font_big.render("COMBO!", True, (255,215,0))
                bg    = pygame.Surface((badge.get_width()+20, badge.get_height()+10), pygame.SRCALPHA)
                bg.fill((30,30,30,180)); self.screen.blit(bg,(5,5)); self.screen.blit(badge,(15,10))
            lbl    = self.font_big.render(self.label, True, (255,255,255))
            lbl_bg = pygame.Surface((self.size[0], lbl.get_height()+16), pygame.SRCALPHA)
            lbl_bg.fill((0,0,0,160))
            self.screen.blit(lbl_bg, (0, self.size[1]-lbl.get_height()-16))
            self.screen.blit(lbl, (self.size[0]//2-lbl.get_width()//2, self.size[1]-lbl.get_height()-8))
            self.timer -= 1
            if self.timer == 0 and self.audio_playing:
                pygame.mixer.music.stop(); self.audio_playing = False
        else:
            self.screen.fill((15,15,25))
            idle = self.font_idle.render("Show a gesture...", True, (90,90,110))
            self.screen.blit(idle,(self.size[0]//2-idle.get_width()//2, self.size[1]//2-idle.get_height()//2))
        pygame.display.flip()
        return True


# ══════════════════════════════════════════════════════════════
#  HUD RENDERER
# ══════════════════════════════════════════════════════════════
class HUDRenderer:
    METHOD_COLORS = {"ml":(255,180,0), "knn":(180,100,255), "rule":(100,200,255)}

    def draw(self, frame, stable_gesture, confidence, method, hand_count, mode,
             record_name="", record_progress=0, samples_needed=60):
        h, w = frame.shape[:2]
        mode_color = {"DETECT":(0,220,90),"RECORD":(30,90,255)}.get(mode,(180,180,180))
        cv2.putText(frame, f"[{mode}]  {hand_count} hand{'s' if hand_count!=1 else ''}",
                    (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2)
        if stable_gesture not in ("unknown","none"):
            m_color = self.METHOD_COLORS.get(method,(200,200,200))
            cv2.putText(frame, f"{stable_gesture}  {int(confidence*100)}%  [{method}]",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, m_color, 2)
            bar_filled = int(200*confidence)
            cv2.rectangle(frame,(10,68),(210,78),(40,40,40),-1)
            bar_color = (0,220,80) if confidence>0.75 else (0,180,255) if confidence>0.5 else (100,80,200)
            cv2.rectangle(frame,(10,68),(10+bar_filled,78),bar_color,-1)
        else:
            cv2.putText(frame,"No gesture",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.60,(70,70,70),1)
        if mode == "RECORD":
            pct   = int((record_progress/samples_needed)*100)
            cv2.putText(frame,f"Recording '{record_name}'  {record_progress}/{samples_needed}  ({pct}%)",
                        (10,h-58),cv2.FONT_HERSHEY_SIMPLEX,0.52,(60,110,255),2)
            bar_w = int(w*record_progress/samples_needed)
            cv2.rectangle(frame,(10,h-38),(w-10,h-22),(30,30,30),-1)
            cv2.rectangle(frame,(10,h-38),(10+bar_w,h-22),(60,110,255),-1)
        cv2.putText(frame,"R=Record  D=Delete  L=List  Q=Quit",
                    (10,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.38,(90,90,90),1)
        return frame


# ══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════
def main():
    global _custom_db_ref, _ml_model_ref

    log.info("=" * 60)
    log.info("  ISHARA_COFFEE — GESTURE RECOGNITION SYSTEM")
    log.info("  R=Record  D=Delete  L=List  Q=Quit")
    log.info("=" * 60)

    pygame.init()
    pygame.mixer.init()

    cfg = Config()

    # ── Camera permission check ──
    log.info("Checking camera access (index %d)...", cfg.CAMERA_INDEX)
    cap_test = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap_test.isOpened():
        log.error("Camera index %d not accessible.", cfg.CAMERA_INDEX)
        if socketio is not None:
            time.sleep(1)
            socketio.emit("camera_error", {"msg": f"Camera {cfg.CAMERA_INDEX} not found."})
        return
    cap_test.release()
    log.info("Camera OK.")

    if not os.path.exists("hand_landmarker.task"):
        log.error("'hand_landmarker.task' not found. Download from mediapipe-models.")
        return

    mp_base    = python.BaseOptions(model_asset_path="hand_landmarker.task")
    mp_options = vision.HandLandmarkerOptions(
        base_options=mp_base, num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mp_hands = vision.HandLandmarker.create_from_options(mp_options)

    assets     = AssetManager(cfg.BUILTIN_ASSETS, cfg.COMBO_ASSETS, cfg.POPUP_SIZE)
    custom_db  = CustomGestureDB(cfg.CUSTOM_DB_FILE, cfg.KNN_K, cfg.KNN_THRESHOLD)
    ml_model   = MLModel("gesture_model.pkl", "label_encoder.pkl", cfg.ML_THRESHOLD)
    classifier = GestureClassifier(ml_model, custom_db, cfg.FINGER_UP_MARGIN)
    popup      = PopupDisplay(cfg.POPUP_SIZE)
    hud        = HUDRenderer()
    cooldown   = CooldownGate(cfg.COOLDOWN_SECONDS)

    _custom_db_ref = custom_db
    _ml_model_ref  = ml_model

    left_buf  = StabilityBuffer(cfg.STABILITY_FRAMES)
    right_buf = StabilityBuffer(cfg.STABILITY_FRAMES)
    clock     = pygame.time.Clock()

    mode              = "DETECT"
    recording_name    = ""
    recording_samples = []
    record_progress   = 0
    last_method       = "rule"
    last_confidence   = 0.0
    last_landmarks_raw = None
    stable_gesture    = "unknown"

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Failed to open camera after check. Aborting.")
        return
    log.info("Camera opened. Running.")

    fps_counter = 0
    fps_timer   = time.time()

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed.")
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts     = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = mp_hands.detect_for_video(mp_img, ts)

        hand_count    = 0
        left_gesture  = "none"
        right_gesture = "none"
        last_landmarks_raw = None

        if result.hand_landmarks and result.handedness:
            hand_count = len(result.hand_landmarks)
            for i, landmarks in enumerate(result.hand_landmarks):
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])), 4, (0,220,80), -1)
                handedness_label      = result.handedness[i][0].category_name
                is_left               = (handedness_label == "Right")
                gesture, conf, method = classifier.classify(landmarks, handedness_label)
                last_landmarks_raw    = landmarks
                last_confidence       = conf
                last_method           = method
                if is_left: left_gesture  = gesture
                else:       right_gesture = gesture
                wx = int(landmarks[0].x*frame.shape[1]); wy = int(landmarks[0].y*frame.shape[0])
                cv2.putText(frame, f"{'L' if is_left else 'R'}:{gesture}",
                            (wx-15,wy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255,200,0) if is_left else (0,180,255), 2)

        stable_left  = left_buf.update(left_gesture)
        stable_right = right_buf.update(right_gesture)

        triggered_name = triggered_surf = triggered_audio = triggered_label = None
        is_combo = False

        if hand_count == 2:
            combo_key  = f"{stable_left}+{stable_right}"
            combo_name = cfg.COMBO_MAP.get(combo_key)
            if combo_name and stable_left not in ("none","unknown") and stable_right not in ("none","unknown"):
                stable_gesture = combo_name; last_confidence = 1.0; last_method = "rule"
                triggered_name = combo_name; triggered_surf  = assets.get_pygame_surf(combo_name)
                triggered_audio = assets.get_audio(combo_name); triggered_label = assets.get_label(combo_name)
                is_combo = True
        elif hand_count == 1:
            single = stable_left if stable_left not in ("none","unknown",None) else stable_right
            if single and single not in ("none","unknown"):
                stable_gesture  = single; triggered_name  = single
                triggered_surf  = assets.get_pygame_surf(single)
                triggered_audio = assets.get_audio(single); triggered_label = assets.get_label(single)
        else:
            stable_gesture = "unknown"

        # ── Web-triggered recording request ──
        if FLASK_AVAILABLE and _record_request.get("ready") and mode == "DETECT":
            recording_name    = _record_request.pop("name")
            _record_request.clear()
            recording_samples = []; record_progress = 0; mode = "RECORD"
            log.info("Web: recording started for '%s'", recording_name)
            if socketio: socketio.emit("record_started", {"name": recording_name})

        # ── Recording ──
        if mode == "RECORD" and last_landmarks_raw is not None:
            recording_samples.append(CustomGestureDB.extract_features(last_landmarks_raw))
            record_progress += 1
            if socketio:
                socketio.emit("record_progress", {
                    "name": recording_name, "progress": record_progress, "total": cfg.SAMPLES_PER_GESTURE
                })
            if record_progress >= cfg.SAMPLES_PER_GESTURE:
                custom_db.add_samples(recording_name, recording_samples)
                log.info("RECORD: Done recording '%s'.", recording_name)
                if socketio: socketio.emit("record_done", {"name": recording_name})
                recording_samples = []; record_progress = 0; recording_name = ""
                mode = "DETECT"; cooldown.reset()

        # ── Trigger popup + emit ──
        if mode == "DETECT" and triggered_name and cooldown.should_trigger(triggered_name):
            popup_audio = None if socketio is not None else triggered_audio
            if triggered_surf is not None:
                popup.trigger(triggered_surf, triggered_label or triggered_name,
                              cfg.DISPLAY_FRAMES, is_combo, popup_audio)
            else:
                popup.trigger(None, f"CUSTOM: {triggered_name}", cfg.DISPLAY_FRAMES, False)

            log.info("%s TRIGGERED: %s (%d%% via %s)",
                     "[COMBO]" if is_combo else "[SINGLE]",
                     triggered_name, int(last_confidence*100), last_method)

            if socketio is not None:
                try:
                    all_assets  = {**cfg.BUILTIN_ASSETS, **cfg.COMBO_ASSETS}
                    asset_entry = all_assets.get(triggered_name, {})
                    socketio.emit("gesture", {
                        "name":       triggered_name,
                        "label":      triggered_label or triggered_name.upper(),
                        "confidence": round(last_confidence*100, 1),
                        "method":     last_method,
                        "is_combo":   is_combo,
                        "image_url":  "/" + asset_entry.get("image","") if asset_entry.get("image") else "",
                        "audio_url":  "/" + asset_entry.get("audio","") if asset_entry.get("audio") else "",
                    }, namespace='/')
                except Exception as exc:
                    log.error("WS emit error: %s", exc)

        keep_running = popup.tick()
        if not keep_running: break

        frame = hud.draw(frame, stable_gesture, last_confidence, last_method,
                         hand_count, mode, recording_name, record_progress, cfg.SAMPLES_PER_GESTURE)

        cv2.imshow("Hand Gesture System", frame)

        with _frame_lock:
            global _latest_frame
            _latest_frame = frame.copy()

        # ── Real FPS + latency ──
        latency_ms   = int((time.time() - frame_start) * 1000)
        fps_counter += 1
        now          = time.time()
        if now - fps_timer >= 1.0:
            real_fps    = fps_counter / (now - fps_timer)
            fps_counter = 0; fps_timer = now
            with _stats_lock:
                _real_stats.update({"fps": round(real_fps,1), "latency_ms": latency_ms,
                                    "hand_count": hand_count, "mode": mode})
            if socketio: socketio.emit("stats", dict(_real_stats))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and mode == "DETECT":
            cap.release(); cv2.destroyAllWindows()
            name = input("\nGesture name (no spaces): ").strip().lower()
            if name:
                if name in custom_db.db:
                    ow = input(f"'{name}' exists. Overwrite? (y/n): ").strip().lower()
                    if ow != 'y':
                        log.info("Cancelled."); cap = cv2.VideoCapture(cfg.CAMERA_INDEX); continue
                    custom_db.delete(name)
                log.info("Recording '%s' in 3s...", name)
                for i in range(3,0,-1): log.info("  %d...", i); time.sleep(1)
                log.info("Recording!")
                recording_name = name; recording_samples = []; record_progress = 0; mode = "RECORD"
            cap = cv2.VideoCapture(cfg.CAMERA_INDEX)

        elif key == ord('d') and mode == "DETECT":
            if not custom_db.db:
                log.info("No custom gestures saved.")
            else:
                cap.release(); cv2.destroyAllWindows()
                for i, n in enumerate(custom_db.db.keys(), 1):
                    log.info("  %d. %s (%d samples)", i, n, len(custom_db.db[n]))
                name = input("Name to delete: ").strip().lower()
                log.info("Deleted '%s'" if custom_db.delete(name) else "Not found: '%s'", name)
                cap = cv2.VideoCapture(cfg.CAMERA_INDEX)

        elif key == ord('l'):
            listing = custom_db.list_gestures()
            if listing:
                for n, count in listing.items(): log.info("  %s: %d samples", n, count)
            else:
                log.info("No custom gestures recorded yet.")

        elif key == ord('q'):
            break

        clock.tick(cfg.TARGET_FPS)

    cap.release(); pygame.quit(); cv2.destroyAllWindows()
    log.info("Exited cleanly.")


if __name__ == "__main__":
    main()