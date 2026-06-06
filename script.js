// --- CONFIGURATION ---
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH_TOP = 13;
const MOUTH_BOTTOM = 14;
const MOUTH_LEFT = 78;
const MOUTH_RIGHT = 308;
const EYEBROW_LEFT = 105;
const EYEBROW_RIGHT = 334;
const NOSE_TIP = 1;

let DYNAMIC_EAR_THRESHOLD = 0.20; 
let DYNAMIC_EAR_WIDE_THRESHOLD = 0.35;
let DYNAMIC_MAR_THRESHOLD = 0.40;
let DYNAMIC_EYEBROW_THRESHOLD = 50.0; // This will calibrate distance

// Dynamic user settings
let DOT_TIME = 0.4;
let MIN_BLINK_TIME = 0.05;
let LETTER_PAUSE = 1.3;
let WORD_PAUSE = 3.0;
let MASTER_VOLUME = 0.1;

// Advanced settings
let isOnlineAIEnabled = true;
let drawFaceLandmarks = true;
let oscType = "sine";
let speechRate = 0.95;
let selectedVoiceName = "";

let EAR_BUFFER = [];
const BUFFER_SIZE = 2;

// EAR History for Sparkline
let earHistory = [];
const MAX_EAR_HISTORY = 100;

// Accuracy tracking
let total_blinks = 0;
let valid_blinks = 0;

// AI Blink Classifier (k-NN) Data Structures
class KNNClassifier {
    constructor(k = 3) {
        this.k = k;
        this.samples = []; // elements: { x: [f1, f2, f3, f4, f5], y: label }
        this.mins = [];
        this.maxs = [];
    }
    
    addSample(features, label) {
        this.samples.push({ x: features, y: label });
    }
    
    clear() {
        this.samples = [];
        this.mins = [];
        this.maxs = [];
    }
    
    train() {
        if (this.samples.length === 0) return;
        const numFeatures = this.samples[0].x.length;
        this.mins = Array(numFeatures).fill(Infinity);
        this.maxs = Array(numFeatures).fill(-Infinity);
        
        for (const sample of this.samples) {
            for (let i = 0; i < numFeatures; i++) {
                if (sample.x[i] < this.mins[i]) this.mins[i] = sample.x[i];
                if (sample.x[i] > this.maxs[i]) this.maxs[i] = sample.x[i];
            }
        }
    }
    
    normalize(x) {
        if (this.mins.length === 0) return x;
        return x.map((val, i) => {
            const range = this.maxs[i] - this.mins[i];
            return range === 0 ? 0 : (val - this.mins[i]) / range;
        });
    }
    
    classify(x) {
        if (this.samples.length < this.k) return null;
        
        const normX = this.normalize(x);
        
        const distances = this.samples.map(sample => {
            const normSampleX = this.normalize(sample.x);
            let sumSq = 0;
            for (let i = 0; i < normX.length; i++) {
                sumSq += Math.pow(normX[i] - normSampleX[i], 2);
            }
            return { dist: Math.sqrt(sumSq), label: sample.y };
        });
        
        // Sort by distance ascending
        distances.sort((a, b) => a.dist - b.dist);
        
        const nearest = distances.slice(0, this.k);
        const votes = {};
        for (const n of nearest) {
            votes[n.label] = (votes[n.label] || 0) + 1;
        }
        
        let maxVotes = -1;
        let bestLabel = null;
        for (const [label, count] of Object.entries(votes)) {
            if (count > maxVotes) {
                maxVotes = count;
                bestLabel = parseInt(label);
            }
        }
        return bestLabel;
    }
}

// Instantiate ML classifier
let mlClassifier = new KNNClassifier(3);
let isMLClassifierEnabled = false;

// AI Trainer State
let isTrainingMode = false;
let trainingClass = 0; // 0 = Reflex, 1 = Dot, 2 = Dash
let dataset = []; // elements: { features: [...], label: int }
let activeBlinkFrames = [];

const MORSE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z"
};

const WORD_LIST = [
    "HELLO","HI","YES","NO","THANK","YOU","HELP","I","AM","GOOD","BAD",
    "PLEASE","WATER","FOOD","PAIN","MORE","LESS","OKAY","STOP","GO","HAPPY",
    "SAD","TIRED","COLD","HOT","GREAT","WHAT","THERE","FRIEND","EVERYONE",
    "WANT","NEED","LIKE","FEEL","ARE","ABOUT","DO","IS","CAN","TIME","REST",
    "GIVE","OUT","HOME","AGAIN","SOME","NOT","THIS","THAT","SIT","STAND",
    "UP","DOWN","LEFT","RIGHT","NOW","LATER"
];

const LOCAL_BIGRAMS = {
    "HELLO": ["WORLD", "THERE", "FRIEND", "EVERYONE"],
    "I": ["AM", "WANT", "NEED", "LIKE", "GO", "FEEL"],
    "HOW": ["ARE", "ABOUT", "DO", "IS"],
    "ARE": ["YOU", "THEY", "WE"],
    "YOU": ["ARE", "CAN", "WANT", "OKAY"],
    "THANK": ["YOU"],
    "PLEASE": ["HELP", "WATER", "FOOD", "GIVE"],
    "WANT": ["WATER", "FOOD", "MORE", "TO", "GO"],
    "NEED": ["HELP", "WATER", "FOOD", "MORE", "REST"],
    "FEEL": ["COLD", "HOT", "TIRED", "SAD", "HAPPY", "GOOD", "BAD"],
    "AM": ["GOOD", "BAD", "TIRED", "HAPPY", "SAD", "COLD", "HOT", "OKAY"],
    "GO": ["TO", "OUT", "HOME", "STOP"],
    "MORE": ["WATER", "FOOD", "TIME", "PLEASE"],
    "LESS": ["PAIN", "NOISE", "LIGHT"],
    "WHAT": ["IS", "ARE", "ABOUT", "TIME"]
};

// App State
let isCameraRunning = false;
let isCalibrating = false;
let calibrationBuffer = [];
const CALIBRATION_FRAMES = 50; // Roughly 2-3 seconds at ~20fps

let eyeClosed = false;
let blinkStart = null;
let lastBlink = Date.now() / 1000;
let gestureCooldown = 0;
let reflexDetectedTime = 0;

let currentMorse = "";
let finalText = "";
let predictedWord = "";
let currentPredictions = ["", "", ""];

// Audio & Speech Context
let audioCtx = null;
function initAudio() {
    if(!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if(audioCtx.state === 'suspended') audioCtx.resume();
}

function playBeep(duration, freq, type) {
    if(!audioCtx || MASTER_VOLUME === 0) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = oscType;
    osc.frequency.value = freq;
    
    gain.gain.setValueAtTime(MASTER_VOLUME, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
}

// Specific Audio Feedback Profiles
function beepDot() { playBeep(0.15, 800, 'sine'); } // High ping
function beepDash() { playBeep(0.3, 400, 'square'); } // Deep hold
function beepAction() { playBeep(0.2, 600, 'triangle'); } // Gesture success
function beepError() { playBeep(0.4, 200, 'sawtooth'); } // Delete/clear tone

// Voice Synthesizer
let synthVoices = [];
function speakText(text) {
    if(!window.speechSynthesis || !text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speechRate;
    utterance.pitch = 1.0;
    
    if (selectedVoiceName) {
        const voice = synthVoices.find(v => v.name === selectedVoiceName);
        if (voice) utterance.voice = voice;
    }
    window.speechSynthesis.speak(utterance);
}

// DOM Elements
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('output_canvas');
const canvasCtx = canvasEl.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');
const startOverlay = document.getElementById('start-overlay');
const cameraBtn = document.getElementById('camera-btn');
const calibrateBtn = document.getElementById('calibrate-btn');
const clearBtn = document.getElementById('clear-btn');
const morseDisplay = document.getElementById('morse-display');
const textDisplay = document.getElementById('text-display');
const blinksVal = document.getElementById('blinks-val');
const validVal = document.getElementById('valid-val');
const accuracyVal = document.getElementById('accuracy-val');
const morseGrid = document.getElementById('morse-grid');
const actionLog = document.getElementById('action-log');

// Settings Elements
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const dotSlider = document.getElementById('dot-slider');
const letterSlider = document.getElementById('letter-slider');
const volSlider = document.getElementById('vol-slider');
const dotVal = document.getElementById('dot-val');
const letterVal = document.getElementById('letter-val');
const volVal = document.getElementById('vol-val');
const historyLog = document.getElementById('history-log');
const exportBtn = document.getElementById('export-btn');

// New Settings Inputs
const aiModeToggle = document.getElementById('ai-mode-toggle');
const landmarkToggle = document.getElementById('landmark-toggle');
const oscTypeSelect = document.getElementById('osc-type');
const rateSlider = document.getElementById('rate-slider');
const rateVal = document.getElementById('rate-val');
const voiceSelect = document.getElementById('voice-select');
const mlClassifierToggle = document.getElementById('ml-classifier-toggle');

// Training Lab Elements
const trainLabBtn = document.getElementById('train-lab-btn');
const trainingModal = document.getElementById('training-modal');
const closeTrainingBtn = document.getElementById('close-training-btn');
const resetDatasetBtn = document.getElementById('reset-dataset-btn');
const trainModelBtn = document.getElementById('train-model-btn');
const btnClass0 = document.getElementById('select-class-0');
const btnClass1 = document.getElementById('select-class-1');
const btnClass2 = document.getElementById('select-class-2');

// Prediction DOM elements
const predPills = [
    document.getElementById('pred-0'),
    document.getElementById('pred-1'),
    document.getElementById('pred-2')
];

// Log Notifications
function showLog(msg, beepType) {
    actionLog.innerText = msg;
    actionLog.classList.add('show');
    if (beepType === 'action') beepAction();
    if (beepType === 'error') beepError();
    setTimeout(() => actionLog.classList.remove('show'), 2000);
}

// Math Helpers
function getDistance(p1, p2) { return Math.hypot(p1.x - p2.x, p1.y - p2.y); }
function computeAspectRatio(landmarks, indices) {
    const A = getDistance(landmarks[indices[1]], landmarks[indices[5]]);
    const B = getDistance(landmarks[indices[2]], landmarks[indices[4]]);
    const C = getDistance(landmarks[indices[0]], landmarks[indices[3]]);
    return (A + B) / (2.0 * C);
}
function computeMAR(landmarks) {
    const H = getDistance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]);
    const W = getDistance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]);
    return W === 0 ? 0 : H / W;
}
function computeEyebrow(landmarks) {
    const L = getDistance(landmarks[EYEBROW_LEFT], landmarks[NOSE_TIP]);
    const R = getDistance(landmarks[EYEBROW_RIGHT], landmarks[NOSE_TIP]);
    return (L + R) / 2;
}

// Autocorrect (Levenshtein)
function levenshteinDistance(a, b) {
    const matrix = Array(b.length + 1).fill().map(() => Array(a.length + 1).fill(0));
    for (let i = 0; i <= b.length; i++) matrix[i][0] = i;
    for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) matrix[i][j] = matrix[i - 1][j - 1];
            else matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1);
        }
    }
    return matrix[b.length][a.length];
}

function autocorrect(word) {
    if (!word) return word;
    let bestMatch = word;
    let highestScore = 0;
    for (const w of WORD_LIST) {
        if(w === word) return word;
        const dist = levenshteinDistance(word, w);
        const maxLen = Math.max(word.length, w.length);
        const score = 1 - (dist / maxLen);
        if (score > highestScore && score >= 0.55) {
            highestScore = score;
            bestMatch = w;
        }
    }
    return bestMatch;
}

// AI Status Label updater
function updateAIStatusText(text, isOnline) {
    const statusTextEl = document.getElementById('ai-status-text');
    const badgeEl = document.getElementById('ai-status');
    if (statusTextEl) statusTextEl.innerText = text;
    if (badgeEl) {
        if (isOnline) {
            badgeEl.style.color = "var(--cyan)";
            badgeEl.style.borderColor = "rgba(6, 182, 212, 0.2)";
            badgeEl.style.background = "rgba(6, 182, 212, 0.08)";
        } else {
            badgeEl.style.color = "var(--text-muted)";
            badgeEl.style.borderColor = "rgba(156, 163, 175, 0.2)";
            badgeEl.style.background = "rgba(156, 163, 175, 0.08)";
        }
    }
}

// Smart Next-Word & Autocomplete AI Prediction
async function updatePredictions() {
    if (!finalText && !currentMorse) {
        currentPredictions = ["", "", ""];
        predictedWord = "";
        renderPredictions("");
        textDisplay.innerText = finalText;
        return;
    }

    const words = finalText.trim().split(/\s+/);
    let context = "";
    let prefix = "";

    if (finalText.length > 0) {
        if (finalText.endsWith(" ")) {
            context = words[words.length - 1] || "";
            prefix = "";
        } else {
            prefix = words[words.length - 1] || "";
            context = words[words.length - 2] || "";
        }
    }

    if (currentMorse) {
        const previewLetter = MORSE_DICT[currentMorse] || "";
        prefix += previewLetter;
    }

    let suggestions = [];

    if (isOnlineAIEnabled && navigator.onLine) {
        try {
            let url = "";
            if (prefix.length > 0) {
                url = `https://api.datamuse.com/sug?s=${prefix.toLowerCase()}`;
            } else if (context.length > 0) {
                url = `https://api.datamuse.com/words?lc=${context.toLowerCase()}`;
            }

            if (url) {
                updateAIStatusText("Cloud API", true);
                const response = await fetch(url);
                const data = await response.json();
                suggestions = data.map(item => item.word.toUpperCase());
            }
        } catch (e) {
            console.warn("Prediction API failed, using local model:", e);
            updateAIStatusText("Local Fallback", false);
        }
    } else {
        updateAIStatusText("Local Engine", false);
    }

    let localSuggestions = [];
    if (prefix.length > 0) {
        localSuggestions = WORD_LIST.filter(w => w.startsWith(prefix) && w !== prefix);
    } else if (context.length > 0) {
        localSuggestions = LOCAL_BIGRAMS[context] || [];
        if (localSuggestions.length < 3) {
            localSuggestions = localSuggestions.concat(WORD_LIST.slice(0, 8));
        }
    } else {
        localSuggestions = ["I", "HELLO", "HOW", "PLEASE"];
    }

    let merged = [...suggestions, ...localSuggestions];
    merged = merged.map(w => w.toUpperCase().trim())
                   .filter(w => w.length > 0 && w !== prefix);

    const uniqueSuggestions = [];
    for (const w of merged) {
        if (!uniqueSuggestions.includes(w) && w !== prefix) {
            uniqueSuggestions.push(w);
        }
        if (uniqueSuggestions.length >= 3) break;
    }

    while (uniqueSuggestions.length < 3) {
        const filler = WORD_LIST.find(w => !uniqueSuggestions.includes(w) && w !== prefix);
        if (filler) {
            uniqueSuggestions.push(filler);
        } else {
            uniqueSuggestions.push("HELP");
        }
    }

    currentPredictions = uniqueSuggestions.slice(0, 3);
    renderPredictions(prefix);
    
    updateInlinePrediction();
    if (predictedWord) {
        textDisplay.innerHTML = finalText + `<span class="prediction">${predictedWord}</span>`;
    } else {
        textDisplay.innerText = finalText;
    }
}

function updateInlinePrediction() {
    const topPrediction = currentPredictions[0] || "";
    if (!topPrediction) {
        predictedWord = "";
        return;
    }

    if (finalText.length > 0 && !finalText.endsWith(" ")) {
        const words = finalText.trim().split(/\s+/);
        const lastWord = words[words.length - 1] || "";
        if (lastWord.length > 0 && topPrediction.startsWith(lastWord)) {
            predictedWord = topPrediction.substring(lastWord.length);
        } else {
            predictedWord = "";
        }
    } else {
        predictedWord = topPrediction;
    }
}

function renderPredictions(prefix) {
    for (let i = 0; i < 3; i++) {
        const pill = predPills[i];
        if (!pill) continue;
        
        const wordEl = pill.querySelector('.word');
        const wordVal = currentPredictions[i] || "";
        
        if (wordEl) {
            if (wordVal) {
                if (prefix && wordVal.startsWith(prefix)) {
                    wordEl.innerHTML = `<span style="color:#ffffff;">${prefix}</span><span style="color:var(--text-muted);">${wordVal.substring(prefix.length)}</span>`;
                } else {
                    wordEl.innerText = wordVal;
                }
                pill.style.visibility = "visible";
                pill.style.pointerEvents = "auto";
                if (i === 0) pill.classList.add('active');
                else pill.classList.remove('active');
            } else {
                wordEl.innerText = "";
                pill.style.visibility = "hidden";
                pill.style.pointerEvents = "none";
                pill.classList.remove('active');
            }
        }
    }
}

function acceptPrediction(index) {
    const wordToInsert = currentPredictions[index];
    if (!wordToInsert) return;
    
    const words = finalText.trim().split(/\s+/);
    
    if (finalText.length > 0 && !finalText.endsWith(" ")) {
        words[words.length - 1] = wordToInsert;
        finalText = words.join(" ") + " ";
    } else {
        finalText = (finalText.trim() + " " + wordToInsert).trim() + " ";
    }
    
    currentMorse = ""; 
    speakText(wordToInsert);
    
    showLog(`AI autocomplete: "${wordToInsert}"`, 'action');
    updateUI();
}

// UI Rendering
function updateUI() {
    if (isCalibrating) return;
    
    // Morse Sequence Rendering
    if (currentMorse) {
        morseDisplay.innerText = currentMorse;
    } else {
        morseDisplay.innerHTML = '<span class="placeholder">Awaiting blink...</span>';
    }
    
    // Start prediction update cycle
    updatePredictions();
    
    // Metrics updates
    blinksVal.innerText = total_blinks;
    validVal.innerText = valid_blinks;
    const accuracy = total_blinks ? ((valid_blinks / total_blinks) * 100).toFixed(2) : 100;
    accuracyVal.innerText = `${accuracy}%`;
    accuracyVal.style.color = accuracy < 50 ? 'var(--danger)' : 'var(--accent)';
}

function clearState() {
    if (finalText.trim().length > 0) {
        const p = document.createElement('p');
        p.innerText = `[${new Date().toLocaleTimeString('en-US',{hour12:false})}] ${finalText.trim()}`;
        historyLog.prepend(p);
    }

    currentMorse = "";
    finalText = "";
    predictedWord = "";
    currentPredictions = ["", "", ""];
    total_blinks = 0;
    valid_blinks = 0;
    EAR_BUFFER = [];
    eyeClosed = false;
    blinkStart = null;
    lastBlink = Date.now() / 1000;
    updateUI();
}

// Draw running EAR sparkline and telemetry HUD on canvas
function drawHUD(ctx, width, height, ear, mar, eyebrow) {
    // 1. Text Telemetry dashboard
    ctx.fillStyle = "rgba(4, 9, 20, 0.7)";
    ctx.fillRect(12, 12, 185, 90);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(12, 12, 185, 90);
    
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "var(--cyan)";
    ctx.fillText("BLINKLY TELEMETRY", 22, 32);
    
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
    ctx.fillText(`EAR (Eyes):   ${ear.toFixed(3)}`, 22, 50);
    ctx.fillText(`MAR (Mouth):  ${mar.toFixed(3)}`, 22, 66);
    ctx.fillText(`BROW (Raise): ${eyebrow.toFixed(1)}`, 22, 82);
    
    // Status text in top right
    let statusText = "TRACKING";
    let statusColor = "var(--accent)"; 
    if (eyeClosed) {
        statusText = "BLINKING";
        statusColor = "var(--primary)"; 
    } else if (mar > DYNAMIC_MAR_THRESHOLD) {
        statusText = "CLEAR TEXT";
        statusColor = "var(--danger)"; 
    } else if (eyebrow > DYNAMIC_EYEBROW_THRESHOLD) {
        statusText = "SPEAKING";
        statusColor = "var(--accent)"; 
    } else if (ear > DYNAMIC_EAR_WIDE_THRESHOLD) {
        statusText = "AI AUTOPLAY";
        statusColor = "var(--cyan)"; 
    }
    
    ctx.fillStyle = "rgba(4, 9, 20, 0.7)";
    ctx.fillRect(width - 152, 12, 140, 36);
    ctx.strokeRect(width - 152, 12, 140, 36);
    
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(width - 137, 30, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillText(statusText, width - 124, 34);
    
    // 2. Alert notifications on camera
    if (eyeClosed) {
        ctx.strokeStyle = "rgba(99, 102, 241, 0.5)";
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, width - 4, height - 4);
    }
    
    if (mar > DYNAMIC_MAR_THRESHOLD) {
        ctx.fillStyle = "rgba(244, 63, 94, 0.15)";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "var(--danger)";
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, width, height);
        
        ctx.font = "bold 20px 'Outfit', sans-serif";
        ctx.fillStyle = "var(--danger)";
        ctx.textAlign = "center";
        ctx.fillText("CLEAR GESTURE: MOUTH OPEN", width / 2, height / 2);
        ctx.textAlign = "start"; 
    }
    
    if (eyebrow > DYNAMIC_EYEBROW_THRESHOLD && finalText.trim().length > 0 && !finalText.endsWith(". ")) {
        ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "var(--accent)";
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, width, height);
        
        ctx.font = "bold 20px 'Outfit', sans-serif";
        ctx.fillStyle = "var(--accent)";
        ctx.textAlign = "center";
        ctx.fillText("SPEAK GESTURE: BROW RAISED", width / 2, height / 2);
        ctx.textAlign = "start"; 
    }
    
    if (ear > DYNAMIC_EAR_WIDE_THRESHOLD && predictedWord && (Date.now() / 1000 - gestureCooldown > 1.5)) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.15)";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "var(--cyan)";
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, width, height);
        
        ctx.font = "bold 20px 'Outfit', sans-serif";
        ctx.fillStyle = "var(--cyan)";
        ctx.textAlign = "center";
        ctx.fillText("AUTOCOMPLETE: EYES WIDE", width / 2, height / 2);
        ctx.textAlign = "start"; 
    }
    
    // AI Reflex Alert Overlay
    if (Date.now() / 1000 - reflexDetectedTime < 1.0) {
        ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, width, height);
        
        ctx.font = "bold 20px 'Outfit', sans-serif";
        ctx.fillStyle = "#f59e0b";
        ctx.textAlign = "center";
        ctx.fillText("REFLEX EYE BLINK IGNORED (AI)", width / 2, height / 2);
        ctx.textAlign = "start";
    }
    
    // 3. EAR Sparkline Graph along bottom
    const graphHeight = 55;
    const graphY = height - graphHeight - 12;
    const graphX = 12;
    const graphWidth = width - 24;
    
    ctx.fillStyle = "rgba(3, 7, 18, 0.8)";
    ctx.fillRect(graphX, graphY, graphWidth, graphHeight);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);
    
    // Gridlines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const gridY = graphY + (graphHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(graphX, gridY);
        ctx.lineTo(graphX + graphWidth, gridY);
        ctx.stroke();
    }
    
    // Threshold dashed line
    const threshPercent = Math.max(0, Math.min(1, DYNAMIC_EAR_THRESHOLD / 0.45));
    const threshY = graphY + graphHeight - (threshPercent * graphHeight);
    
    ctx.strokeStyle = "rgba(244, 63, 94, 0.5)"; 
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(graphX, threshY);
    ctx.lineTo(graphX + graphWidth, threshY);
    ctx.stroke();
    ctx.setLineDash([]); 
    
    ctx.fillStyle = "rgba(244, 63, 94, 0.85)";
    ctx.font = "bold 9px 'JetBrains Mono', monospace";
    ctx.fillText(`THRESHOLD CLAMP: ${DYNAMIC_EAR_THRESHOLD.toFixed(3)}`, graphX + 8, threshY - 5);
    
    // Plot historical EAR lines
    if (earHistory.length > 1) {
        ctx.strokeStyle = "var(--cyan)"; 
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        
        for (let i = 0; i < earHistory.length; i++) {
            const val = earHistory[i];
            const percent = Math.max(0, Math.min(1, val / 0.45));
            const ptX = graphX + (graphWidth / (MAX_EAR_HISTORY - 1)) * i;
            const ptY = graphY + graphHeight - (percent * graphHeight);
            
            if (i === 0) {
                ctx.moveTo(ptX, ptY);
            } else {
                ctx.lineTo(ptX, ptY);
            }
        }
        ctx.stroke();
    }
}

// --- Feature Extraction ---
function extractBlinkFeatures(frames, duration, threshold) {
    if (frames.length === 0) frames = [threshold * 0.5];
    const min_ear = Math.min(...frames);
    const min_index = frames.indexOf(min_ear);
    
    const depth = Math.max(0, threshold - min_ear);
    
    let max_drop = 0;
    for (let i = 0; i < min_index; i++) {
        const diff = frames[i] - frames[i+1];
        if (diff > max_drop) max_drop = diff;
    }
    
    let max_rise = 0;
    for (let i = min_index; i < frames.length - 1; i++) {
        const diff = frames[i+1] - frames[i];
        if (diff > max_rise) max_rise = diff;
    }
    
    return {
        duration: duration,
        min_ear: min_ear,
        depth: depth,
        max_drop: max_drop,
        max_rise: max_rise
    };
}

// --- FaceMesh Process ---
if (!window.FaceMesh) console.warn("FaceMesh library missing from window object.");
else {
    window.faceMesh = new FaceMesh({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`});
    faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    faceMesh.onResults(onResults);
}

function triggerMorseAnimation() {
    morseDisplay.classList.remove('morse-added');
    void morseDisplay.offsetWidth; 
    morseDisplay.classList.add('morse-added');
}

function triggerReflexOverlay() {
    reflexDetectedTime = Date.now() / 1000;
}

// Intercept blink classifier and commit Morse code
function processBlinkEvent(featureVector, duration) {
    if (isMLClassifierEnabled && mlClassifier.samples.length >= 3) {
        const prediction = mlClassifier.classify(featureVector);
        
        if (prediction === 0) {
            triggerReflexOverlay();
            showLog("[AI: Reflex Ignored]", "action");
        } else if (prediction === 1) {
            currentMorse += ".";
            beepDot();
            triggerMorseAnimation();
            updateUI();
        } else if (prediction === 2) {
            currentMorse += "-";
            beepDash();
            triggerMorseAnimation();
            updateUI();
        }
    } else {
        if (duration < DOT_TIME) {
            currentMorse += ".";
            beepDot();
        } else {
            currentMorse += "-";
            beepDash();
        }
        triggerMorseAnimation();
        updateUI();
    }
}

// Record dataset sample in Training Lab Mode
function recordTrainingSample(featureVector) {
    const classCount = dataset.filter(item => item.label === trainingClass).length;
    if (classCount >= 15) {
        showTrainingConsole(`Class ${trainingClass} limit reached! (Max 15 samples)`);
        beepError();
        return;
    }
    
    dataset.push({ features: featureVector, label: trainingClass });
    beepAction();
    
    updateTrainingUI();
    drawTrainingPlot();
    
    showTrainingConsole(`Recorded Class ${trainingClass} Sample #${classCount + 1}: Dur=${featureVector[0].toFixed(2)}s, Depth=${featureVector[2].toFixed(2)}`);
}

function onResults(results) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        const currentTime = Date.now() / 1000;
        const width = canvasEl.width;
        const height = canvasEl.height;
        
        const scaledLandmarks = landmarks.map(lm => ({ x: lm.x * width, y: lm.y * height }));
        
        const leftEAR = computeAspectRatio(scaledLandmarks, LEFT_EYE);
        const rightEAR = computeAspectRatio(scaledLandmarks, RIGHT_EYE);
        const rawEAR = (leftEAR + rightEAR) / 2;
        
        const mar = computeMAR(scaledLandmarks);
        const eyebrow = computeEyebrow(scaledLandmarks);
        
        earHistory.push(rawEAR);
        if (earHistory.length > MAX_EAR_HISTORY) {
            earHistory.shift();
        }
        
        // --- 1. Personal Auto-Calibration Core Logic ---
        if (isCalibrating) {
            calibrationBuffer.push({ear: rawEAR, mar: mar, eyebrow: eyebrow});
            canvasCtx.fillStyle = "rgba(16, 185, 129, 0.3)";
            canvasCtx.fillRect(0,0,width,height);
            
            const scanY = (Math.sin(Date.now() / 150) + 1) / 2 * height;
            canvasCtx.strokeStyle = "rgba(16, 185, 129, 0.8)";
            canvasCtx.lineWidth = 3;
            canvasCtx.beginPath();
            canvasCtx.moveTo(0, scanY);
            canvasCtx.lineTo(width, scanY);
            canvasCtx.stroke();
            
            if (calibrationBuffer.length >= CALIBRATION_FRAMES) {
                const restingEAR = calibrationBuffer.reduce((a,b)=>a+b.ear,0)/CALIBRATION_FRAMES;
                const restingMAR = calibrationBuffer.reduce((a,b)=>a+b.mar,0)/CALIBRATION_FRAMES;
                const restingEyebrow = calibrationBuffer.reduce((a,b)=>a+b.eyebrow,0)/CALIBRATION_FRAMES;
                
                DYNAMIC_EAR_THRESHOLD = restingEAR * 0.82; 
                DYNAMIC_EAR_WIDE_THRESHOLD = restingEAR * 1.25; 
                DYNAMIC_MAR_THRESHOLD = restingMAR + 0.09; 
                DYNAMIC_EYEBROW_THRESHOLD = restingEyebrow * 1.08; 
                
                isCalibrating = false;
                showLog(`Calibrated: EAR ${DYNAMIC_EAR_THRESHOLD.toFixed(2)}`, 'action');
                updateUI();
            }
            canvasCtx.restore();
            return;
        }

        // --- Visual Face Tracking Dots ---
        if (drawFaceLandmarks) {
            canvasCtx.fillStyle = "rgba(6, 182, 212, 0.5)";
            [...LEFT_EYE, ...RIGHT_EYE, MOUTH_TOP, MOUTH_BOTTOM, EYEBROW_LEFT, EYEBROW_RIGHT].forEach(i => {
                canvasCtx.beginPath();
                canvasCtx.arc(scaledLandmarks[i].x, scaledLandmarks[i].y, 2.5, 0, 2 * Math.PI);
                canvasCtx.fill();
            });
        }

        // --- 2. Advanced Multi-Gestural Face Controls ---
        if (currentTime - gestureCooldown > 1.5) {
            if (mar > DYNAMIC_MAR_THRESHOLD) {
                clearState();
                showLog("MOUTH OPENED: Text Cleared", "error");
                gestureCooldown = currentTime;
            } 
            else if (eyebrow > DYNAMIC_EYEBROW_THRESHOLD && finalText.trim().length > 0) {
                const sentence = finalText.trim();
                if (!finalText.endsWith(". ")) {
                    showLog("BROWS RAISED: Voiced Sentence", "action");
                    speakText(sentence);
                    finalText += ". "; 
                    updateUI();
                }
                gestureCooldown = currentTime;
            }
        }
        
        // --- 3. EAR Blink Processing ---
        EAR_BUFFER.push(rawEAR);
        if (EAR_BUFFER.length > BUFFER_SIZE) EAR_BUFFER.shift();
        const ear = EAR_BUFFER.reduce((a, b) => a + b, 0) / EAR_BUFFER.length;
        
        if (ear > DYNAMIC_EAR_WIDE_THRESHOLD && currentTime - gestureCooldown > 1.5) {
            if (predictedWord) {
                acceptPrediction(0);
                gestureCooldown = currentTime;
            }
        }
        
        if (ear < DYNAMIC_EAR_THRESHOLD && !eyeClosed) {
            eyeClosed = true;
            blinkStart = currentTime;
            activeBlinkFrames = [rawEAR];
            canvasCtx.fillStyle = "rgba(99, 102, 241, 0.3)";
            canvasCtx.fillRect(0, 0, width, height);
            
        } else if (eyeClosed) {
            activeBlinkFrames.push(rawEAR);
            
            if (ear >= DYNAMIC_EAR_THRESHOLD) {
                eyeClosed = false;
                const duration = currentTime - blinkStart;
                lastBlink = currentTime;
                total_blinks++;
                
                if (duration >= MIN_BLINK_TIME) {
                    const features = extractBlinkFeatures(activeBlinkFrames, duration, DYNAMIC_EAR_THRESHOLD);
                    const featureVector = [features.duration, features.min_ear, features.depth, features.max_drop, features.max_rise];
                    
                    if (isTrainingMode) {
                        recordTrainingSample(featureVector);
                    } else {
                        valid_blinks++;
                        processBlinkEvent(featureVector, duration);
                    }
                }
            }
        }
        
        // Render HUD display details
        drawHUD(canvasCtx, width, height, rawEAR, mar, eyebrow);
    }
    canvasCtx.restore();
}

// Core Polling Loop for completing Letters and Words
setInterval(() => {
    if (!isCameraRunning || isCalibrating) return;
    const currentTime = Date.now() / 1000;
    const pause = currentTime - lastBlink;
    let changed = false;
    
    // Complete letter on pause
    if (!eyeClosed && pause > LETTER_PAUSE && currentMorse) {
        const letter = MORSE_DICT[currentMorse] || "?";
        finalText += letter;
        currentMorse = "";
        lastBlink = currentTime; 
        changed = true;
    }
    
    // Complete word and apply spelling autocorrect
    if (!eyeClosed && pause > WORD_PAUSE && finalText && !finalText.endsWith(" ") && !finalText.endsWith(". ")) {
        const words = finalText.trim().split(" ");
        const lastWordRaw = words[words.length - 1];
        
        const corrected = autocorrect(lastWordRaw);
        words[words.length - 1] = corrected;
        finalText = words.join(" ") + " ";
        
        speakText(corrected);
        changed = true;
    }
    
    if (changed) updateUI();
}, 100);

let camera = null;

async function toggleCamera() {
    initAudio(); 
    if (isCameraRunning) {
        if (camera) camera.stop();
        isCameraRunning = false;
        cameraBtn.innerText = "Start Camera";
        cameraBtn.classList.replace('secondary', 'primary');
        calibrateBtn.disabled = true;
        
        const labBtn = document.getElementById('train-lab-btn');
        if (labBtn) labBtn.disabled = true;
        
        startOverlay.style.display = 'flex';
        loadingOverlay.style.display = 'none';
        canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    } else {
        startOverlay.style.display = 'none';
        loadingOverlay.style.display = 'flex';
        cameraBtn.disabled = true;
        cameraBtn.innerText = "Starting...";
        if (!camera) {
            camera = new Camera(videoEl, {
                onFrame: async () => {
                    if (isCameraRunning) {
                        try { await faceMesh.send({image: videoEl}); } catch(err) { console.error(err); }
                    }
                },
                width: 640, height: 480
            });
        }
        await camera.start();
        isCameraRunning = true;
        loadingOverlay.style.display = 'none';
        cameraBtn.innerText = "Pause Camera";
        cameraBtn.classList.replace('primary', 'secondary');
        cameraBtn.disabled = false;
        calibrateBtn.disabled = false;
        
        const labBtn = document.getElementById('train-lab-btn');
        if (labBtn) labBtn.disabled = false;
        
        lastBlink = Date.now() / 1000;
        
        showLog("System Ready. Please Calibrate!", "action");
    }
}

calibrateBtn.addEventListener('click', () => {
    isCalibrating = true;
    calibrationBuffer = [];
    currentMorse = "";
    
    morseDisplay.innerHTML = '<span class="placeholder" style="color:var(--cyan); font-weight:800;">[ CALIBRATING ] Look straight at camera...</span>';
    beepAction();
});

cameraBtn.addEventListener('click', toggleCamera);
clearBtn.addEventListener('click', clearState);

// --- Settings Modals & Listeners ---
settingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('hidden');
});
closeSettingsBtn.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
});
dotSlider.addEventListener('input', (e) => {
    DOT_TIME = parseFloat(e.target.value);
    dotVal.innerText = `${DOT_TIME.toFixed(2)}s`;
});
letterSlider.addEventListener('input', (e) => {
    LETTER_PAUSE = parseFloat(e.target.value);
    letterVal.innerText = `${LETTER_PAUSE.toFixed(1)}s`;
});
volSlider.addEventListener('input', (e) => {
    MASTER_VOLUME = parseFloat(e.target.value);
    volVal.innerText = `${Math.round(MASTER_VOLUME / 0.5 * 100)}%`;
});

// Dynamic Settings Integrations
if (aiModeToggle) {
    aiModeToggle.addEventListener('change', (e) => {
        isOnlineAIEnabled = e.target.checked;
        updatePredictions();
    });
}
if (landmarkToggle) {
    landmarkToggle.addEventListener('change', (e) => {
        drawFaceLandmarks = e.target.checked;
    });
}
if (oscTypeSelect) {
    oscTypeSelect.addEventListener('change', (e) => {
        oscType = e.target.value;
    });
}
if (rateSlider) {
    rateSlider.addEventListener('input', (e) => {
        speechRate = parseFloat(e.target.value);
        rateVal.innerText = `${speechRate.toFixed(2)}x`;
    });
}

// Speech voices loader
function populateVoiceList() {
    if (!window.speechSynthesis) return;
    synthVoices = window.speechSynthesis.getVoices();
    if (voiceSelect) {
        voiceSelect.innerHTML = '<option value="">System Default Voice</option>';
        synthVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.name;
            option.innerText = `${voice.name} (${voice.lang})`;
            voiceSelect.appendChild(option);
        });
    }
}
if (window.speechSynthesis) {
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = populateVoiceList;
    }
    populateVoiceList();
}
if (voiceSelect) {
    voiceSelect.addEventListener('change', (e) => {
        selectedVoiceName = e.target.value;
    });
}

// AI Training Lab Event Listeners
if (trainLabBtn) {
    trainLabBtn.addEventListener('click', () => {
        trainingModal.classList.remove('hidden');
        isTrainingMode = true;
        updateTrainingUI();
        setTimeout(drawTrainingPlot, 100); 
    });
}

if (closeTrainingBtn) {
    closeTrainingBtn.addEventListener('click', () => {
        trainingModal.classList.add('hidden');
        isTrainingMode = false;
    });
}

function setTrainingClass(cls) {
    trainingClass = cls;
    [btnClass0, btnClass1, btnClass2].forEach((btn, idx) => {
        if (btn) {
            if (idx === cls) btn.classList.add('active');
            else btn.classList.remove('active');
        }
    });
    for (let i = 0; i < 3; i++) {
        const card = document.getElementById(`step-card-${i}`);
        if (card) {
            if (i === cls) card.classList.add('active');
            else card.classList.remove('active');
        }
    }
    showTrainingConsole(`Switched recording mode to Class ${cls}`);
    beepAction();
}

if (btnClass0) btnClass0.addEventListener('click', () => setTrainingClass(0));
if (btnClass1) btnClass1.addEventListener('click', () => setTrainingClass(1));
if (btnClass2) btnClass2.addEventListener('click', () => setTrainingClass(2));

if (resetDatasetBtn) {
    resetDatasetBtn.addEventListener('click', () => {
        dataset = [];
        mlClassifier.clear();
        isMLClassifierEnabled = false;
        if (mlClassifierToggle) mlClassifierToggle.checked = false;
        updateClassifierStatusUI();
        updateTrainingUI();
        drawTrainingPlot();
        showTrainingConsole("Dataset reset successfully.");
        beepError();
    });
}

if (trainModelBtn) {
    trainModelBtn.addEventListener('click', () => {
        mlClassifier.clear();
        dataset.forEach(item => {
            mlClassifier.addSample(item.features, item.label);
        });
        mlClassifier.train();
        
        isMLClassifierEnabled = true;
        if (mlClassifierToggle) mlClassifierToggle.checked = true;
        
        updateClassifierStatusUI();
        
        trainingModal.classList.add('hidden');
        isTrainingMode = false;
        
        showLog("AI Model Activated", "action");
        beepAction();
    });
}

if (mlClassifierToggle) {
    mlClassifierToggle.addEventListener('change', (e) => {
        if (e.target.checked && mlClassifier.samples.length < 3) {
            showLog("AI Classifier is not trained!", "error");
            e.target.checked = false;
            isMLClassifierEnabled = false;
        } else {
            isMLClassifierEnabled = e.target.checked;
            updateClassifierStatusUI();
            showLog(isMLClassifierEnabled ? "ML Classifier ON" : "ML Classifier OFF", "action");
        }
    });
}

function updateClassifierStatusUI() {
    const textEl = document.getElementById('classifier-status-text');
    const badgeEl = document.getElementById('classifier-status-badge');
    
    if (textEl) {
        if (isMLClassifierEnabled) {
            textEl.innerText = "ML: Active";
        } else {
            if (mlClassifier.samples.length >= 3) {
                textEl.innerText = "ML: Ready";
            } else {
                textEl.innerText = "ML: Off";
            }
        }
    }
    
    if (badgeEl) {
        if (isMLClassifierEnabled) {
            badgeEl.style.color = "var(--cyan)";
            badgeEl.style.borderColor = "rgba(6, 182, 212, 0.4)";
            badgeEl.style.background = "rgba(6, 182, 212, 0.15)";
        } else if (mlClassifier.samples.length >= 3) {
            badgeEl.style.color = "var(--text-main)";
            badgeEl.style.borderColor = "rgba(255, 255, 255, 0.2)";
            badgeEl.style.background = "rgba(255, 255, 255, 0.05)";
        } else {
            badgeEl.style.color = "var(--text-muted)";
            badgeEl.style.borderColor = "rgba(156, 163, 175, 0.25)";
            badgeEl.style.background = "rgba(156, 163, 175, 0.05)";
        }
    }
}

function checkDatasetStatus() {
    const reflexCount = dataset.filter(item => item.label === 0).length;
    const dotCount = dataset.filter(item => item.label === 1).length;
    const dashCount = dataset.filter(item => item.label === 2).length;
    
    const trainBtn = document.getElementById('train-model-btn');
    const statusText = document.getElementById('dataset-status');
    
    if (reflexCount >= 5 && dotCount >= 5 && dashCount >= 5) {
        if (trainBtn) trainBtn.disabled = false;
        if (statusText) {
            statusText.innerText = "Ready to Train";
            statusText.className = "status-success";
        }
    } else {
        if (trainBtn) trainBtn.disabled = true;
        if (statusText) {
            statusText.innerText = "Untrained (Need 5+ each)";
            statusText.className = "status-warning";
        }
    }
}

function updateTrainingUI() {
    for (let i = 0; i < 3; i++) {
        const count = dataset.filter(item => item.label === i).length;
        const pct = Math.min(100, (count / 10) * 100);
        
        const bar = document.getElementById(`progress-${i}`);
        const text = document.getElementById(`progress-text-${i}`);
        
        if (bar) bar.style.width = `${pct}%`;
        if (text) text.innerText = `${count} / 10`;
    }
    checkDatasetStatus();
}

function showTrainingConsole(msg) {
    const consoleEl = document.getElementById('training-console');
    if (consoleEl) {
        consoleEl.innerText = msg;
    }
}

// Draw Training Scatter Plot on Canvas
function drawTrainingPlot() {
    const canvas = document.getElementById('training_canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth;
    const h = canvas.height = canvas.parentElement.clientHeight || 200;
    
    ctx.clearRect(0, 0, w, h);
    
    const pad = 35;
    const graphW = w - pad * 2;
    const graphH = h - pad * 2;
    
    // Draw axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();
    
    // Axis ticks / labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText("0.0s", pad - 10, h - pad + 15);
    ctx.fillText("1.0s (Duration)", w - pad - 60, h - pad + 15);
    
    ctx.save();
    ctx.translate(pad - 15, pad + 70);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Depth (EAR)", 0, 0);
    ctx.restore();
    
    // Gridlines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
        const x = pad + (graphW / 5) * i;
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, h - pad);
        ctx.stroke();
    }
    for (let i = 1; i <= 4; i++) {
        const y = pad + (graphH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
    }
    
    // Plot points
    dataset.forEach(item => {
        const dur = item.features[0];
        const depth = item.features[2];
        
        // Map values: Dur 0 to 1.0s, Depth 0 to 0.4
        const x = pad + Math.min(1.0, Math.max(0.0, dur / 1.0)) * graphW;
        const y = h - pad - Math.min(1.0, Math.max(0.0, depth / 0.4)) * graphH;
        
        let color = "#f59e0b"; // Class 0
        if (item.label === 1) color = "var(--cyan)"; // Class 1
        if (item.label === 2) color = "var(--primary)"; // Class 2
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
    });
}

function initSettings() {
    if(aiModeToggle) isOnlineAIEnabled = aiModeToggle.checked;
    if(landmarkToggle) drawFaceLandmarks = landmarkToggle.checked;
    if(oscTypeSelect) oscType = oscTypeSelect.value;
    if(rateSlider) {
        speechRate = parseFloat(rateSlider.value);
        rateVal.innerText = `${speechRate.toFixed(2)}x`;
    }
    initPredictionListeners();
}

// Setup Keyboard shortcuts to choose prediction options (Keys 1, 2, 3)
document.addEventListener('keydown', (e) => {
    if (settingsModal && !settingsModal.classList.contains('hidden')) return;
    if (trainingModal && !trainingModal.classList.contains('hidden')) return;
    
    if (e.key === '1') {
        acceptPrediction(0);
    } else if (e.key === '2') {
        acceptPrediction(1);
    } else if (e.key === '3') {
        acceptPrediction(2);
    }
});

function initPredictionListeners() {
    predPills.forEach((pill, idx) => {
        if (!pill) return;
        pill.addEventListener('click', () => {
            acceptPrediction(idx);
        });
    });
}

// Export logs
exportBtn.addEventListener('click', () => {
    if(!historyLog.innerText.trim()) {
        showLog("History is empty!", "error");
        return;
    }
    const blob = new Blob([historyLog.innerText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Blinkly_Chat_Log_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showLog("History Exported!", "action");
});

// Initialize Morse Grid list
function initMorseDictionary() {
    if (!morseGrid) return;
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    const invertedDict = {};
    for (const [code, letter] of Object.entries(MORSE_DICT)) invertedDict[letter] = code;
    alphabet.forEach(letter => {
        const code = invertedDict[letter];
        if (code) {
            const div = document.createElement('div');
            div.className = 'morse-item';
            div.innerHTML = `<span class="morse-char">${letter}</span><span class="morse-code">${code}</span>`;
            morseGrid.appendChild(div);
        }
    });
}

initMorseDictionary();
initSettings();
updateUI();
updateClassifierStatusUI();
