"""
BrailleAI - Braille Character Recognition System
Hybrid Deep Learning Framework: CNN + BiLSTM-CTC
With Text-to-Speech Audio Output
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from gtts import gTTS
import json
import os
import io

# ============================================================================
# MODEL DEFINITIONS (must match training code exactly)
# ============================================================================

class BrailleCNN(nn.Module):
    """CNN for single Braille character classification (A-Z)."""
    def __init__(self, num_classes=26):
        super(BrailleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrailleCRNN(nn.Module):
    """CNN-BiLSTM-CTC model for Braille sentence recognition."""
    def __init__(self, num_classes, img_height=48, hidden_size=256, num_lstm_layers=2):
        super(BrailleCRNN, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        cnn_output_height = 3
        rnn_input_size = 512 * cnn_output_height
        self.linear_bridge = nn.Linear(rnn_input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_lstm_layers,
                           batch_first=True, bidirectional=True,
                           dropout=0.3 if num_lstm_layers > 1 else 0)
        self.output = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        conv = self.dropout(self.linear_bridge(conv))
        rnn_out, _ = self.rnn(conv)
        output = self.output(rnn_out).permute(1, 0, 2)
        return F.log_softmax(output, dim=2)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load both models and their metadata."""
    device = torch.device('cpu')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Load character model
    char_checkpoint = torch.load(os.path.join(models_dir, 'char_model_full.pth'),
                                  map_location=device, weights_only=False)
    char_meta = char_checkpoint['metadata']
    char_model = BrailleCNN(num_classes=char_meta['num_classes'])
    char_model.load_state_dict(char_checkpoint['model_state_dict'])
    char_model.eval()

    # Load sentence model
    sent_checkpoint = torch.load(os.path.join(models_dir, 'sent_model_full.pth'),
                                  map_location=device, weights_only=False)
    sent_meta = sent_checkpoint['metadata']
    sent_model = BrailleCRNN(
        num_classes=sent_meta['num_classes'],
        img_height=sent_meta['img_height'],
        hidden_size=sent_meta['hidden_size'],
        num_lstm_layers=sent_meta['num_lstm_layers'],
    )
    sent_model.load_state_dict(sent_checkpoint['model_state_dict'])
    sent_model.eval()

    return char_model, char_meta, sent_model, sent_meta, device


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_character(model, image, metadata, device):
    """Predict a single Braille character."""
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    img = image.convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    idx_to_class = metadata['idx_to_class']
    predicted_class = idx_to_class[str(predicted.item())] if str(predicted.item()) in idx_to_class else idx_to_class.get(predicted.item(), '?')

    # Get top 5 predictions
    top5_probs, top5_indices = probabilities.topk(5, dim=1)
    top5 = []
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        cls = idx_to_class.get(str(idx), idx_to_class.get(idx, '?'))
        top5.append((cls, prob))

    return predicted_class, confidence.item(), top5


def predict_sentence(model, image, metadata, device):
    """Predict a Braille sentence using CTC decoding."""
    img_height = metadata['img_height']
    img_max_width = metadata['img_max_width']
    idx_to_char = metadata['idx_to_char']

    img = image.convert('L')
    w, h = img.size
    new_h = img_height
    new_w = int(w * (new_h / h))
    new_w = min(new_w, img_max_width)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new('L', (img_max_width, img_height), color=0)
    padded.paste(img, (0, 0))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img_tensor = transform(padded).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Greedy CTC decode
    _, max_indices = output.max(2)
    indices = max_indices.squeeze(1).cpu().numpy()

    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:
            ch_key = str(idx)
            ch = idx_to_char.get(ch_key, idx_to_char.get(idx, '?'))
            chars.append(ch)
        prev_idx = idx

    predicted_text = ''.join(chars)

    # Get confidence
    probs = torch.exp(output).squeeze(1)
    max_probs, _ = probs.max(1)
    non_blank_mask = max_indices.squeeze(1).cpu() != 0
    if non_blank_mask.any():
        avg_confidence = max_probs[non_blank_mask].mean().item()
    else:
        avg_confidence = 0.0

    return predicted_text, avg_confidence


# ============================================================================
# TEXT-TO-SPEECH FUNCTION
# ============================================================================

def text_to_audio(text):
    """Convert text to audio bytes using gTTS."""
    if not text or text.strip() == "":
        return None
    try:
        # For single characters, spell them out clearly
        if len(text.strip()) == 1:
            speak_text = f"The character is {text}"
        else:
            speak_text = text

        tts = gTTS(text=speak_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Page config
st.set_page_config(
    page_title="BrailleAI",
    page_icon="⠃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark BrailleAI theme
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a14 100%);
        color: #e0e0e0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #6ee7b7 40%, #818cf8 70%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .main-header .subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #6ee7b7;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .main-header .description {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #8b8fa3;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(110, 231, 183, 0.1);
        border: 1px solid rgba(110, 231, 183, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6ee7b7;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #6ee7b7;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Mode selector tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 0;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
        max-width: 400px;
        margin: 0 auto 2rem auto;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        color: #8b8fa3;
        border-radius: 10px;
        padding: 10px 24px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #818cf8, #6366f1) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Upload area */
    .stFileUploader > div {
        background: rgba(255,255,255,0.02);
        border: 2px dashed rgba(129, 140, 248, 0.3);
        border-radius: 16px;
        padding: 2rem;
    }
    .stFileUploader > div:hover {
        border-color: rgba(129, 140, 248, 0.6);
        background: rgba(129, 140, 248, 0.03);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(129, 140, 248, 0.08), rgba(110, 231, 183, 0.05));
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #6ee7b7;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .result-value-sentence {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
        word-break: break-word;
    }
    .result-confidence {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #8b8fa3;
    }
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #6ee7b7, #818cf8);
    }

    /* Top predictions */
    .top-pred {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .top-pred-char {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #e0e0e0;
    }
    .top-pred-bar-container {
        flex-grow: 1;
        margin: 0 1rem;
        height: 4px;
        background: rgba(255,255,255,0.05);
        border-radius: 2px;
        overflow: hidden;
    }
    .top-pred-bar {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #818cf8, #6ee7b7);
    }
    .top-pred-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #6ee7b7;
        min-width: 55px;
        text-align: right;
    }

    /* Audio section */
    .audio-section {
        background: rgba(110, 231, 183, 0.05);
        border: 1px solid rgba(110, 231, 183, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }
    .audio-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6ee7b7;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Model info footer */
    .model-info {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .model-info-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6ee7b7;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .model-info-grid {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .model-info-item {
        text-align: center;
    }
    .model-info-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #ffffff;
    }
    .model-info-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        color: #8b8fa3;
    }

    /* Image container */
    .uploaded-image-container {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:center; margin-bottom:1rem;">
        <div class="status-badge">
            <div class="status-dot"></div>
            HYBRID ML FRAMEWORK · ACTIVE
        </div>
    </div>
    <h1>Braille Character<br>Recognition System</h1>
    <p class="description">
        Upload a Braille image to recognise individual characters or full sentences
        using a two-phase hybrid machine learning pipeline. Results include text output
        and audio playback for accessibility.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load Models ──
try:
    char_model, char_meta, sent_model, sent_meta, device = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Failed to load models. Make sure model files are in the `models/` folder.\n\nError: {e}")

if models_loaded:
    # ── Mode Tabs ──
    tab_char, tab_sent = st.tabs(["⠁  Single Character", "⠿  Full Sentence"])

    # ── Tab 1: Single Character ──
    with tab_char:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("##### Upload Braille Character")
            char_file = st.file_uploader(
                "Upload a single Braille character image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="char_upload",
                label_visibility="collapsed",
            )

            if char_file is not None:
                image = Image.open(char_file)
                st.markdown('<div class="uploaded-image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if char_file is not None:
                with st.spinner("Analyzing..."):
                    predicted_class, confidence, top5 = predict_character(
                        char_model, image, char_meta, device
                    )

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Predicted Character</div>
                    <div class="result-value">{predicted_class}</div>
                    <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Audio output
                st.markdown('<div class="audio-section"><div class="audio-label">Audio Output</div></div>', unsafe_allow_html=True)
                audio_data = text_to_audio(predicted_class)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### Top 5 Predictions")
                for cls, prob in top5:
                    st.markdown(f"""
                    <div class="top-pred">
                        <span class="top-pred-char">{cls}</span>
                        <div class="top-pred-bar-container">
                            <div class="top-pred-bar" style="width: {prob*100}%"></div>
                        </div>
                        <span class="top-pred-pct">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card" style="opacity: 0.5;">
                    <div class="result-label">Awaiting Input</div>
                    <div class="result-value" style="font-size:2rem;">⠁</div>
                    <div class="result-confidence">Upload a Braille character image to begin</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: Full Sentence ──
    with tab_sent:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("##### Upload Braille Sentence")
            sent_file = st.file_uploader(
                "Upload a Braille sentence image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="sent_upload",
                label_visibility="collapsed",
            )

            if sent_file is not None:
                image = Image.open(sent_file)
                st.markdown('<div class="uploaded-image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if sent_file is not None:
                with st.spinner("Decoding sentence..."):
                    predicted_text, confidence = predict_sentence(
                        sent_model, image, sent_meta, device
                    )

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Decoded Sentence</div>
                    <div class="result-value-sentence">{predicted_text if predicted_text else '(empty)'}</div>
                    <div class="result-confidence">Average Confidence: {confidence*100:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Audio output
                if predicted_text:
                    st.markdown('<div class="audio-section"><div class="audio-label">Audio Output — Listen to Translation</div></div>', unsafe_allow_html=True)
                    audio_data = text_to_audio(predicted_text)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### Character Breakdown")
                    chars_display = '  →  '.join(list(predicted_text.replace(' ', '␣')))
                    st.code(chars_display, language=None)

                    # Download option
                    st.download_button(
                        label="📥 Download as Text File",
                        data=predicted_text,
                        file_name="braille_translation.txt",
                        mime="text/plain",
                    )
            else:
                st.markdown("""
                <div class="result-card" style="opacity: 0.5;">
                    <div class="result-label">Awaiting Input</div>
                    <div class="result-value" style="font-size:2rem;">⠿</div>
                    <div class="result-confidence">Upload a Braille sentence image to begin</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Model Info Footer ──
    st.markdown(f"""
    <div class="model-info">
        <div class="model-info-title">Model Architecture</div>
        <div class="model-info-grid">
            <div class="model-info-item">
                <div class="model-info-value">CNN</div>
                <div class="model-info-label">Character Model</div>
            </div>
            <div class="model-info-item">
                <div class="model-info-value">BiLSTM-CTC</div>
                <div class="model-info-label">Sentence Model</div>
            </div>
            <div class="model-info-item">
                <div class="model-info-value">26</div>
                <div class="model-info-label">Character Classes</div>
            </div>
            <div class="model-info-item">
                <div class="model-info-value">{char_meta.get('test_accuracy', 0):.1f}%</div>
                <div class="model-info-label">Char Accuracy</div>
            </div>
            <div class="model-info-item">
                <div class="model-info-value">{sent_meta.get('test_word_accuracy', 0):.1f}%</div>
                <div class="model-info-label">Sentence Accuracy</div>
            </div>
            <div class="model-info-item">
                <div class="model-info-value">gTTS</div>
                <div class="model-info-label">Audio Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
