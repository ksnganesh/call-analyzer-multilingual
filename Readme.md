# 🎧 Multilingual Call Sentiment Analyzer (Local MVP)

A fully local FastAPI-based app that analyzes call recordings across Indian languages (English, Hindi, Tamil, Telugu, Kannada, Malayalam).

## 🚀 Features
- 🎙 Transcription via Whisper (multilingual)
- 🧍 Speaker Diarization using Resemblyzer
- ❤️ Sentiment & Emotion scoring with multilingual model (`tabularisai/multilingual-sentiment-analysis`)
- 🧠 LLM-based Summaries & Action Suggestions (OpenAI API)
- 📊 Scoring Rubric for Telecaller & Customer
- 🌐 Web Interface to upload audio and view results

## 🛠 Setup
```bash
git clone https://github.com/ksnganesh/call-analyzer-multilingual.git
cd call-analyzer-multilingual
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt