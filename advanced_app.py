import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import winsound
import threading
import pyttsx3
import difflib
import os
from datetime import datetime

# --- SYSTEM SETTINGS ---
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
BROW_LEFT, BROW_RIGHT, NOSE_TIP = 105, 334, 1

DYN_EAR_THRESH = 0.20
DYN_EAR_WIDE_THRESH = 0.35
DYN_MAR_THRESH = 0.40
DYN_BROW_THRESH = 50.0

DOT_TIME = 0.4
MIN_BLINK_TIME = 0.05
LETTER_PAUSE = 1.3
WORD_PAUSE = 3.0
BUFFER_SIZE = 2

MORSE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z"
}

WORD_LIST = ["HELLO","HI","YES","NO","THANK","YOU","HELP","I","AM","GOOD","BAD",
             "PLEASE","WATER","FOOD","PAIN","MORE","LESS","OKAY","STOP","GO","HAPPY"]

# --- UTILITIES ---
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_ear(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def autocorrect(word):
    if not word: return word
    matches = difflib.get_close_matches(word.upper(), WORD_LIST, n=1, cutoff=0.55)
    return matches[0] if matches else word

class AIPredictor:
    def __init__(self):
        self.model_loaded = False
        self.pipeline = None
        self.current_context = ""
        self.latest_prediction = ""
        self.loading_msg = "Loading AI NLP..."
        self._lock = threading.Lock()
        
        # Load in background so UI frame rate isn't impacted
        threading.Thread(target=self._load_model, daemon=True).start()
        
    def _load_model(self):
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from transformers import pipeline
            import torch
            
            # Using distilgpt2 for very fast next-word context inference on CPU
            self.pipeline = pipeline("text-generation", model="distilgpt2", pad_token_id=50256)
            self.model_loaded = True
            self.loading_msg = ""
            print("AI NLP Model loaded successfully.")
        except ImportError:
            self.loading_msg = "[pip install torch transformers]"
        except Exception as e:
            self.loading_msg = f"AI Error"
            print(f"Error loading AI model: {e}")

    def request_prediction(self, text):
        if not self.model_loaded or not text:
            with self._lock:
                self.latest_prediction = ""
            return
            
        with self._lock:
            if text == self.current_context:
                return 
            self.current_context = text
            self.latest_prediction = ""
            
        threading.Thread(target=self._predict, args=(text,), daemon=True).start()
        
    def _predict(self, text):
        try:
            # Keep context small for faster CPU inference
            prompt = text[-60:] 
            
            result = self.pipeline(
                prompt,
                max_new_tokens=2, 
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                truncation=True
            )
            
            generated = result[0]['generated_text']
            new_text = generated[len(prompt):]
            
            pred = ""
            if new_text:
                words = new_text.replace('\n', ' ').split()
                if words:
                    pred = words[0].strip(".,!?\"'()").upper()
                    
            with self._lock:
                if self.current_context == text:
                    self.latest_prediction = pred
        except Exception as e:
            pass
            
    def get_prediction(self):
        if not self.model_loaded:
            return self.loading_msg
        with self._lock:
            return self.latest_prediction

ai_predictor = AIPredictor()

# Audio and Voice
engine = pyttsx3.init()
def speak(text):
    if text:
        def run_speech():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run_speech, daemon=True).start()

def beep(freq, duration_ms):
    threading.Thread(target=lambda: winsound.Beep(freq, duration_ms), daemon=True).start()

# --- MAIN APP ---
class AdvancedBlinkly:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Blinkly")
        self.root.configure(bg="#0f172a") # Dark Slate background
        self.root.geometry("800x850")

        # State Variables
        self.is_calibrating = False
        self.cal_buffer = []
        self.ear_buffer = []
        
        self.eye_closed = False
        self.blink_start = None
        self.last_blink = time.time()
        self.gesture_cooldown = 0
        
        # Online ML State
        self.rest_ear = 0.30
        self.current_blink_min_ear = 1.0
        
        self.current_morse = ""
        self.final_text = ""
        self.predicted = ""
        self.total_blinks = 0
        self.valid_blinks = 0
        self.history = []

        # UI Setup
        self.setup_ui()

        # MediaPipe & Camera Setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)

        # Start Loop
        self.update_loop()

    def setup_ui(self):
        # Header Controls
        ctrl_frame = tk.Frame(self.root, bg="#1e293b", pady=15, padx=15)
        ctrl_frame.pack(fill=tk.X)
        
        tk.Button(ctrl_frame, text="Auto-Calibrate Face", command=self.start_calibrate, 
                  bg="#3b82f6", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
                  
        tk.Button(ctrl_frame, text="Clear Text", command=self.clear_state, 
                  bg="#ef4444", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
                  
        tk.Button(ctrl_frame, text="Export History", command=self.export_log, 
                  bg="#10b981", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT).pack(side=tk.RIGHT, padx=5)

        # Video Label
        video_container = tk.Frame(self.root, bg="#000", padx=5, pady=5)
        video_container.pack(pady=15)
        self.video_lbl = tk.Label(video_container, bg="black")
        self.video_lbl.pack()

        # Data Frame
        data_frame = tk.Frame(self.root, bg="#1e293b", pady=15, padx=25)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.morse_lbl = tk.Label(data_frame, text="Morse: ", font=("Courier", 24, "bold"), fg="#3b82f6", bg="#1e293b")
        self.morse_lbl.pack(anchor="w", pady=5)

        self.pred_lbl = tk.Label(data_frame, text="Prediction: ", font=("Arial", 14, "italic"), fg="#94a3b8", bg="#1e293b")
        self.pred_lbl.pack(anchor="w")

        self.text_lbl = tk.Label(data_frame, text="Text: ", font=("Arial", 22, "bold"), fg="#f8fafc", bg="#1e293b", wraplength=700, justify="left")
        self.text_lbl.pack(anchor="w", pady=10)

        self.acc_lbl = tk.Label(data_frame, text="Accuracy: 100%", font=("Arial", 12), fg="#10b981", bg="#1e293b")
        self.acc_lbl.pack(anchor="e")

        self.log_lbl = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="#ef4444", bg="#0f172a")
        self.log_lbl.pack(pady=5)

    def log(self, msg, color="#10b981"):
        self.log_lbl.config(text=msg, fg=color)
        self.root.after(3000, lambda: self.log_lbl.config(text=""))

    def start_calibrate(self):
        self.is_calibrating = True
        self.cal_buffer = []
        self.log("CALIBRATING: Keep face neutral and look at camera...", "#f59e0b")
        beep(600, 200)

    def clear_state(self):
        if self.final_text.strip():
            self.history.append(f"[{datetime.now().strftime('%H:%M:%S')}] {self.final_text.strip()}")
            
        self.current_morse = ""
        self.final_text = ""
        self.predicted = ""
        self.total_blinks = 0
        self.valid_blinks = 0
        self.ear_buffer = []
        self.eye_closed = False
        self.last_blink = time.time()
        self.update_ui()

    def export_log(self):
        if not self.history:
            messagebox.showinfo("Wait", "No history to export.")
            return
        filename = f"BlinkTranslatorLog_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write("\n".join(self.history))
        self.log(f"Exported to {filename}", "#10b981")
        beep(600, 200)

    def update_ui(self):
        self.morse_lbl.config(text=f"Morse: {self.current_morse}" if self.current_morse else "Morse: (waiting)")
        self.pred_lbl.config(text=f"Autofill: {self.predicted}" if self.predicted else "")
        self.text_lbl.config(text=f"Text: {self.final_text}")
        
        acc = (self.valid_blinks / self.total_blinks * 100) if self.total_blinks else 100
        self.acc_lbl.config(text=f"Accuracy: {acc:.2f}%", fg="#10b981" if acc >= 50 else "#ef4444")

    def update_loop(self):
        global DYN_EAR_THRESH, DYN_EAR_WIDE_THRESH, DYN_MAR_THRESH, DYN_BROW_THRESH
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_loop)
            return
            
        frame = cv2.flip(frame, 1) # Mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        current_time = time.time()
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])
            
            # --- Geometric Extraction ---
            left_eye = coords[LEFT_EYE_IDXS]
            right_eye = coords[RIGHT_EYE_IDXS]
            mouth_vert = coords[[MOUTH_TOP, MOUTH_BOTTOM]]
            mouth_horz = coords[[MOUTH_LEFT, MOUTH_RIGHT]]
            brow_left, brow_right, nose = coords[BROW_LEFT], coords[BROW_RIGHT], coords[NOSE_TIP]
            
            raw_ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2
            mar = distance(mouth_vert[0], mouth_vert[1]) / (distance(mouth_horz[0], mouth_horz[1]) + 1e-6)
            eyebrow = (distance(brow_left, nose) + distance(brow_right, nose)) / 2
            
            # --- Visual Tracking Drawing ---
            cv2.polylines(frame, [left_eye], True, (0, 200, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 200, 0), 1)
            cv2.circle(frame, tuple(mouth_vert[0]), 2, (0, 0, 255), -1)
            cv2.circle(frame, tuple(mouth_vert[1]), 2, (0, 0, 255), -1)

            # --- 1. Calibration ---
            if self.is_calibrating:
                self.cal_buffer.append((raw_ear, mar, eyebrow))
                # Soft green overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                if len(self.cal_buffer) >= 50:
                    ears, mars, brows = zip(*self.cal_buffer)
                    self.rest_ear = sum(ears)/50
                    rest_mar = sum(mars)/50
                    rest_brow = sum(brows)/50
                    
                    DYN_EAR_THRESH = self.rest_ear * 0.85
                    DYN_EAR_WIDE_THRESH = rest_ear * 1.30
                    DYN_MAR_THRESH = rest_mar + 0.08
                    DYN_BROW_THRESH = rest_brow * 1.10
                    
                    self.is_calibrating = False
                    self.log(f"Calibrated successfully!", "#10b981")
                    beep(600, 200)
            else:
                # --- 2. Advanced Multi-Gestures ---
                if current_time - self.gesture_cooldown > 1.5:
                    # Clear Text
                    if mar > DYN_MAR_THRESH: 
                        self.clear_state()
                        self.log("MOUTH OPEN: Cleared Text", "#ef4444")
                        beep(200, 400)
                        self.gesture_cooldown = current_time
                        
                    # Speak/Space
                    elif eyebrow > DYN_BROW_THRESH and self.final_text.strip(): 
                        if not self.final_text.endswith(". "):
                            speak(self.final_text.strip())
                            self.final_text += ". "
                            self.log("BROWS RAISED: Spoken text", "#3b82f6")
                            self.update_ui()
                        self.gesture_cooldown = current_time
                        
                # --- 3. Robust EAR Processing ---
                self.ear_buffer.append(raw_ear)
                if len(self.ear_buffer) > BUFFER_SIZE: self.ear_buffer.pop(0)
                smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)
                
                # Autocomplete Wide Eyes Gesture
                if smooth_ear > DYN_EAR_WIDE_THRESH and current_time - self.gesture_cooldown > 1.5:
                    if self.predicted and self.predicted not in ["Loading AI NLP...", "[pip install torch transformers]", "AI Error"]:
                        self.final_text += self.predicted + " "
                        self.current_morse = ""
                        speak(self.final_text.strip().split()[-1])
                        self.log("EYES WIDE: Autocompleted AI Word", "#10b981")
                        self.predicted = ""
                        self.update_ui()
                        self.gesture_cooldown = current_time
                
                # Blink Detection
                if smooth_ear < DYN_EAR_THRESH and not self.eye_closed:
                    self.eye_closed = True
                    self.blink_start = current_time
                    self.current_blink_min_ear = smooth_ear
                    
                    # Blue tint on closed eyes
                    cv2.rectangle(frame, (0,0), (w,h), (255, 0, 0), 10) 
                    
                elif self.eye_closed:
                    self.current_blink_min_ear = min(self.current_blink_min_ear, smooth_ear)
                    
                    if smooth_ear >= DYN_EAR_THRESH:
                        self.eye_closed = False
                        dur = current_time - self.blink_start
                        self.last_blink = current_time
                        self.total_blinks += 1
                        
                        if dur >= MIN_BLINK_TIME:
                            # Disabled strict Natural Reflex Classifier because it was rejecting valid faint 'Dots'
                            is_reflex = False 
                            
                            if is_reflex:
                                pass
                            else:
                                self.valid_blinks += 1
                                if dur < DOT_TIME:
                                    self.current_morse += "."
                                    beep(800, 150)
                                else:
                                    self.current_morse += "-"
                                    beep(400, 300)
                                self.update_ui()
                
                # Fatigue Adaptation (Online Learning)
                # ONLY adapt if the eye is resting normally (not pulled wide open)
                elif (DYN_EAR_THRESH * 1.1) < smooth_ear < (self.rest_ear * 1.05) and current_time - self.last_blink > 1.0:
                    # Slower, safer EMA adaptation
                    self.rest_ear = (self.rest_ear * 0.9995) + (smooth_ear * 0.0005)
                    DYN_EAR_THRESH = self.rest_ear * 0.85
                    DYN_EAR_WIDE_THRESH = self.rest_ear * 1.30
                            
        # --- 4. Logic Polling ---
        pause = current_time - self.last_blink
        changed = False
        
        if not self.is_calibrating:
            if not self.eye_closed and pause > LETTER_PAUSE and self.current_morse:
                letter = MORSE_DICT.get(self.current_morse, "?")
                self.final_text += letter
                self.current_morse = ""
                self.last_blink = current_time
                changed = True
                
            if not self.eye_closed and pause > WORD_PAUSE and self.final_text and not self.final_text.endswith(" ") and not self.final_text.endswith(". "):
                words = self.final_text.strip().split(" ")
                corrected = autocorrect(words[-1])
                words[-1] = corrected
                self.final_text = " ".join(words) + " "
                speak(corrected)
                changed = True
                
            ai_predictor.request_prediction(self.final_text)
            new_pred = ai_predictor.get_prediction()
            if new_pred != self.predicted:
                self.predicted = new_pred
                changed = True

        if changed:
            self.update_ui()

        # Render Frame to Tkinter Label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        from PIL import ImageTk
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_lbl.imgtk = imgtk
        self.video_lbl.configure(image=imgtk)
        
        # Loop
        self.root.after(10, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBlinkly(root)
    root.mainloop()
