# OmniTranslate — Real-Time Voice Translation

A fully local, serverless-hybrid voice translation system that runs **English ↔ Japanese** in your browser.  
All processing stays on your machine — no cloud, no API keys, no data leaves your device.

## How It Works

```
Microphone
   │
   ▼
[Browser] AudioWorklet VAD          ← detects when you stop speaking
   │
   ▼
[Browser] Whisper-Tiny (WASM/WebGPU) ← speech → text  (~500ms)
   │  transcribed text
   ▼
[Local Server] SMaLL-100             ← translation    (~200ms)
   │  translated text (shown immediately)
   ▼
[Local Server] OmniVoice TTS         ← text → voice   (~45s CPU)
   │  Base64 WAV audio
   ▼
[Browser] Web Audio API              ← plays the translated voice
```

## Requirements

- Python 3.10+ with pip
- Required packages (install once):

```powershell
pip install transformers torch soundfile omnivoice huggingface_hub
```

- The `tokenization_small100.py` file must stay in the same folder as `server.py` (custom SMaLL-100 tokenizer).
- Models are downloaded automatically from Hugging Face on first run and cached locally.

## Starting the App

**1. Start the server** (in PowerShell or Command Prompt, from this folder):

```powershell
C:\Users\nived\AppData\Local\Python\bin\python.exe server.py
```

> The first time you run this, the server needs to load two AI models into memory.  
> **SMaLL-100** loads in ~5 seconds. **OmniVoice** takes **3–5 minutes** on CPU — this is normal.  
> Do not close the terminal. Wait until you see:
> ```
> ************************************************************
> *  SERVER IS READY — open http://localhost:8000           *
> ************************************************************
> ```

**2. Open your browser** and go to:

```
http://localhost:8000
```

**3. Use the app:**

1. Click **Initialize Models** — Whisper-Tiny downloads/loads into your browser (~30s first time, instant after).
2. Once the status dot turns **green** and shows `Listening...`, click **Start Listening**.
3. Allow microphone permissions if prompted.
4. **Speak clearly** in English (or Japanese — use the toggle to switch direction).
5. Pause for ~1 second — the VAD will detect your silence and trigger transcription.
6. The **transcribed text** and **translation** appear within ~1 second.
7. The **OmniVoice audio** plays ~45 seconds later (CPU synthesis time).

## Files

| File | Purpose |
|---|---|
| `server.py` | Python HTTP server — serves the web app + runs MT and TTS models |
| `index.html` | Web app UI |
| `index.css` | Styling |
| `app.js` | Browser JS — loads Whisper, handles VAD events, fetches translation/TTS |
| `audio-worklet.js` | AudioWorklet thread — Voice Activity Detection (VAD) |
| `tokenization_small100.py` | Custom tokenizer for the SMaLL-100 translation model |

## Stopping the Server

Press `Ctrl+C` in the terminal.

## Notes

- **TTS latency on CPU:** OmniVoice takes ~45 seconds per utterance on CPU. The translation text appears immediately; audio plays when ready.
- **First run:** Model weights (~2GB total) are downloaded once and cached in `~/.cache/huggingface/`.
- **Privacy:** All speech, transcription, and translation happens entirely on your local machine.
