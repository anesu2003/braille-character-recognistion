# BrailleAI — Braille Character Recognition System

A hybrid deep learning framework for recognising Braille characters and sentences, built with **CNN** and **BiLSTM-CTC** architectures.

## Features

- **Single Character Recognition** — Upload a Braille character image and get instant A-Z classification with confidence scores
- **Full Sentence Recognition** — Upload a Braille sentence image and get the decoded English text using CNN-BiLSTM-CTC pipeline
- **Dark-themed UI** — Clean, accessible web interface built with Streamlit

## Architecture

| Component | Model | Task |
|-----------|-------|------|
| Phase 1 | CNN (3-block, 32→64→128) | Single character classification (A-Z) |
| Phase 2 | CNN-BiLSTM-CTC (4-block CNN + 2-layer BiLSTM) | Sentence/word recognition |

## Project Structure

```
braille-ai/
├── app.py                  # Streamlit application
├── models/
│   ├── char_model_full.pth # Trained CNN character model
│   ├── sent_model_full.pth # Trained CRNN sentence model  
│   └── model_config.json   # Model metadata
├── .streamlit/
│   └── config.toml         # Theme configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/braille-ai.git
cd braille-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Live Demo

🔗 [Deployed on Streamlit Cloud](https://your-app-url.streamlit.app)

## Training

Models were trained on:
- **Character dataset**: 2,600 images (100 per class, A-Z), 50×50 grayscale
- **Sentence dataset**: 2,000 synthetic Braille sentence images with text labels

Training was performed on Kaggle with GPU acceleration.

## Tech Stack

- PyTorch (CNN, LSTM, CTC Loss)
- Streamlit (Web UI)
- Python, PIL, torchvision

## Author

Peter Mundowa
