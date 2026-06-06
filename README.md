# 👁️ Blinkly - AI Morse & Gesture Dashboard

An advanced, premium, hands-free web dashboard that empowers users to communicate using only their facial movements. Blinkly leverages computer vision to translate eye blinks into Morse Code strings instantly, backed by a cloud-connected AI text prediction engine and an adaptive browser-side Machine Learning blink classifier.

---

## ✨ Features

1. **Intelligent Face Mesh Tracking:** Uses `MediaPipe Face Mesh` to track eye aspect ratio (EAR) and facial gestures. All camera processing is done locally, keeping your camera stream 100% private.
2. **AI Blink Classifier & Involuntary Reflex Filter (Local ML):**
   - **Personalized AI Training Lab:** Collect your custom dataset directly in the browser. Record 10 natural/reflex blinks (involuntary), 10 intentional dots (short deliberate blinks), and 10 intentional dashes (long deliberate blinks).
   - **Interactive Scatter Plot:** Visualizes your recorded blink dataset in 2D space (Duration vs. Depth) using HTML5 Canvas in real-time.
   - **Client-Side k-NN Classifier:** Trains an adaptive **k-Nearest Neighbors (k-NN, k=3)** model on your custom dataset. Once enabled, involuntary reflex blinks are filtered out (`[REFLEX IGNORED]` flashes on HUD) while deliberate inputs translate flawlessly.
3. **Next-Gen Hybrid AI Word Prediction:** 
   - **Online Mode:** Queries the **Datamuse API** for real-time word completions (while typing) and context-aware next-word predictions (after typing a space).
   - **Offline Fallback:** Automatically drops back to a local trigram/bigram frequency model if the internet connection is lost.
   - **Pill Bar Selection:** Displays the top 3 suggestions. Accept the primary suggestion with the "Eyes Wide Open" gesture, click the pills directly, or use keyboard shortcuts `1`, `2`, or `3`.
4. **Futuristic HUD Overlay & Live Telemetry:**
   - Visual corner brackets, scanner lines, and crosshairs for a sci-fi cockpit look.
   - A live **Eye Aspect Ratio (EAR) EKG Sparkline Graph** drawn directly on the canvas feed showing your blink signals and calibration thresholds.
   - Dynamic canvas alerts that flash when gestures are active (e.g. mouth open, brow raise, reflex filter trigger).
5. **Multi-Gestural Action Controls:**
   - **Wide Mouth Open:** Instantly clears current sentence and text.
   - **Raised Eyebrows:** Inserts a period, ends the sentence, and speaks the text out loud.
   - **Eyes Wide Open:** Accepts the active AI predicted word suggestion.
6. **Speech Synthesizer & Audio Feedback:**
   - Outputs customized audio tones: a high-pitched dot (`.`) and a lower-pitched dash (`-`).
   - Configurable oscillator tone waveforms (Sine, Triangle, Square, Sawtooth) and custom speech rate speeds in the Settings Panel.
   - Select customized text-to-speech accents and voices (Web Speech API) directly from a settings dropdown.
7. **Smart Personal Calibration:** Mathematical baseline sampling for custom eye structures to adapt to user fatigue over time.
8. **History Log Exporting:** Easily export local logs of translated chat sessions directly to a `.txt` file.

---

## 🚀 How to Run

### Option A: Web Version (Zero Setup - Recommended)
1. Simply open [index.html](file:///c:/Users/rian2/.gemini/antigravity-ide/scratch/kerala-student-jobs/blink-translator/index.html) in any modern browser (Chrome, Edge, Safari, Firefox).
2. Allow webcam permission.
3. Click **Start Camera**, calibrate your resting face by clicking **Calibrate Face**, and start blinking!
4. Open the **Training Lab** to record your blink dataset and train your customized ML filter!

No installation or local server required!

### Option B: Advanced Desktop Version (Python)
To run the native desktop version (`advanced_app.py`):
1. Install Python dependencies:
   ```bash
   pip install opencv-python mediapipe numpy Pillow pyttsx3
   ```
2. Run the application:
   ```bash
   python advanced_app.py
   ```

---

## 📖 Morse Code Quick Guide

- Short blink (< 0.4s) = **Dot** (`.`)
- Long blink (> 0.4s) = **Dash** (`-`)
- Pause 1.3s = **Letter break** (combines current dots/dashes into a letter)
- Pause 3.0s = **Word break** (applies autocorrect, voices the word, and requests next-word predictions)
