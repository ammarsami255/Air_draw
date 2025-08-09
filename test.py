import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
import time

# =============================
#       CONFIGURATION
# =============================

RIGHT_MODEL_PATH = "right_model.p"
LEFT_MODEL_PATH = "left_model.p"
SCREENSHOT_DIR = "screenshots"

BRUSH_THICKNESS = 6
ERASER_THICKNESS = 50
FONT_SIZE = 0.7
COLORS = [
    (0, 0, 255),     # أحمر
    (0, 255, 0),     # أخضر
    (255, 0, 0),     # أزرق
    (0, 255, 255),   # أصفر
    (255, 0, 255),   # بنفسجي
    (255, 255, 255)  # أبيض
]

DEBUG_MODE = True
RIGHT_MAP = {"0": "draw"}
LEFT_MAP = {
    "0": "save",
    "1": "pen",
    "2": "color_mode",  # دلوقتي معناها تفعيل قائمة الألوان
    "3": "erase",
    "4": "bigger_brush",
    "5": "smaller_brush"
}

COLOR_CHANGE_DELAY = 0.5

def load_model(path):
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and "model" in model_data:
        return model_data["model"]
    return model_data

def extract_features(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    data = []
    for lm in hand_landmarks.landmark:
        data.append(lm.x - min(x_))
        data.append(lm.y - min(y_))
    return np.array(data).reshape(1, -1)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class HandDrawingTool:
    def __init__(self):
        self.right_model = load_model(RIGHT_MODEL_PATH)
        self.left_model = load_model(LEFT_MODEL_PATH)
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.current_mode = "pen"
        self.color_idx = 0
        self.last_color_change_time = 0
        self.show_color_menu = False
        ensure_dir(SCREENSHOT_DIR)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label
                features = extract_features(hand_landmarks)

                if hand_label == "Right":
                    pred = self.right_model.predict(features)[0]
                    action = RIGHT_MAP.get(pred)
                    self.handle_right_hand(action, hand_landmarks, w, h)
                else:
                    pred = self.left_model.predict(features)[0]
                    action = LEFT_MAP.get(pred)
                    self.handle_left_hand(action)

                if DEBUG_MODE:
                    cv2.putText(frame, f"{hand_label}: {pred}",
                                (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)

        frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

        if self.show_color_menu:
            self.draw_color_menu(frame)

        return frame

    def handle_right_hand(self, action, hand_landmarks, w, h):
        x = int(hand_landmarks.landmark[8].x * w)
        y = int(hand_landmarks.landmark[8].y * h)

        if self.show_color_menu:
            self.check_color_selection(x, y)
            self.prev_x, self.prev_y = None, None
            return

        if action == "draw":
            if self.prev_x is None or self.prev_y is None:
                self.prev_x, self.prev_y = x, y
            if self.current_mode == "pen":
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y),
                         COLORS[self.color_idx], BRUSH_THICKNESS)
            elif self.current_mode == "erase":
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y),
                         (0, 0, 0), ERASER_THICKNESS)
            self.prev_x, self.prev_y = x, y
        else:
            self.prev_x, self.prev_y = None, None

    def handle_left_hand(self, action):
        if action == "pen":
            self.prev_x, self.prev_y = None, None
            self.current_mode = "pen"
            self.show_color_menu = False
        elif action == "color_mode":
            self.show_color_menu = True
        elif action == "erase":
            self.prev_x, self.prev_y = None, None
            self.current_mode = "erase"
            self.show_color_menu = False
        elif action == "save":
            self.save_screenshot()
        elif action == "bigger_brush":
            self.change_brush_size(2)
        elif action == "smaller_brush":
            self.change_brush_size(-2)

    def change_brush_size(self, delta):
        global BRUSH_THICKNESS, ERASER_THICKNESS
        BRUSH_THICKNESS = max(1, BRUSH_THICKNESS + delta)
        ERASER_THICKNESS = max(5, ERASER_THICKNESS + delta * 5)

    def save_screenshot(self):
        filename = os.path.join(SCREENSHOT_DIR,
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, self.canvas)
        print(f"[INFO] Screenshot saved: {filename}")

    def draw_color_menu(self, frame):
        h, w, _ = frame.shape
        box_size = 60
        spacing = 10
        start_x = 10
        y = 10
        for i, color in enumerate(COLORS):
            x1 = start_x + i * (box_size + spacing)
            x2 = x1 + box_size
            cv2.rectangle(frame, (x1, y), (x2, y + box_size), color, -1)
            if i == self.color_idx:
                cv2.rectangle(frame, (x1, y), (x2, y + box_size), (255, 255, 255), 2)

    def check_color_selection(self, x, y):
        box_size = 60
        spacing = 10
        start_x = 10
        y_box = 10
        if y_box <= y <= y_box + box_size:
            for i in range(len(COLORS)):
                x1 = start_x + i * (box_size + spacing)
                x2 = x1 + box_size
                if x1 <= x <= x2:
                    now = time.time()
                    if now - self.last_color_change_time > COLOR_CHANGE_DELAY:
                        self.color_idx = i
                        self.show_color_menu = False
                        self.last_color_change_time = now
                        print(f"[INFO] Color changed to {COLORS[i]}")

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            frame = self.process_frame(frame)

            cv2.putText(frame, f"Mode: {self.current_mode} | Brush: {BRUSH_THICKNESS}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 2)

            cv2.imshow("Hand Drawing Tool", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('+'):
                self.change_brush_size(2)
            elif key == ord('-'):
                self.change_brush_size(-2)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = HandDrawingTool()
    tool.run()
