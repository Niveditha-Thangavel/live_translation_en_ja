import os
import sys
import io
import json
import time
import base64
import wave
from http.server import SimpleHTTPRequestHandler, HTTPServer
import torch
import numpy as np

# Bypass Windows symlink privilege requirements
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load HF_TOKEN from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed — fall back to reading .env manually
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get('HF_TOKEN'):
    print("ERROR: HF_TOKEN not found. Create a .env file with:\n  HF_TOKEN=your_token_here")
    sys.exit(1)

# Force UTF-8 for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("Starting Hybrid Server...")

# ── 1. Translation Model (SMaLL-100) ──────────────────────────────────────────
print("[1/2] Loading Translation model (SMaLL-100)...")
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer

mt_model     = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to("cpu")
mt_tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
print("[1/2] Translation model loaded!\n")

# ── 2. OmniVoice TTS ──────────────────────────────────────────────────────────
print("[2/2] Loading OmniVoice TTS model (this takes ~60–90 s)...")
from omnivoice import OmniVoice
tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cpu",
    dtype=torch.float32,
    load_asr=False   # Skip ASR weights — browser handles STT
)

# Force-flush so this always appears even after garbled progress bars
sys.stdout.flush()
print("\n")
print("*" * 60)
print("*  SERVER IS READY — open http://localhost:8000           *")
print("*" * 60 + "\n", flush=True)


# ── Audio helpers ──────────────────────────────────────────────────────────────
def audio_list_to_wav_b64(audio_list, sample_rate=24000):
    """
    Convert OmniVoice output (list of arrays/tensors) → Base64 WAV string.
    Follows the same pattern as test_omnivoice.py: concatenate all chunks first.
    """
    # Concatenate all chunks (OmniVoice returns a list of segments)
    data = np.concatenate([np.asarray(a, dtype=np.float32) for a in audio_list])
    data = np.squeeze(data)

    if data.ndim != 1 or data.size == 0:
        raise ValueError(f"Unexpected audio shape after concat+squeeze: {data.shape}")

    print(f"[TTS] Audio: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")

    # Normalize amplitude
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.9

    # Encode as 16-bit PCM WAV in memory
    pcm = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    print(f"[TTS] WAV → Base64: {len(b64)} chars  ({data.size / sample_rate:.2f}s audio)")
    return b64


def send_json(handler, data, status=200):
    body = json.dumps(data).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    handler.wfile.write(body)


# ── HTTP Handler ───────────────────────────────────────────────────────────────
class HybridHandler(SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Only log POST requests; suppress noisy GET static-file chatter
        if self.command == 'POST':
            print(f"[{self.command}] {self.path}  {args[1]}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length   = int(self.headers.get('Content-Length', 0))
        raw_body = self.rfile.read(length)

        try:
            req = json.loads(raw_body.decode('utf-8'))
        except Exception:
            send_json(self, {"error": "Invalid JSON"}, status=400)
            return

        # ── /translate  ── fast MT only, returns text immediately ──────────────
        if self.path == '/translate':
            text     = req.get('text', '').strip()
            src_lang = req.get('src_lang', 'en')
            tgt_lang = 'ja' if src_lang == 'en' else 'en'

            if not text:
                send_json(self, {"translation": "", "metrics": {"mt_time_ms": 0}})
                return

            print(f"\n[MT]  '{text}'  ({src_lang} → {tgt_lang})")
            t0 = time.time()

            mt_tokenizer.src_lang = src_lang
            mt_tokenizer.tgt_lang = tgt_lang
            inputs  = mt_tokenizer(text, return_tensors="pt")
            tokens  = mt_model.generate(
                **inputs,
                forced_bos_token_id=mt_tokenizer.get_lang_id(tgt_lang)
            )
            translated = mt_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            mt_ms = int((time.time() - t0) * 1000)
            print(f"[MT]  → '{translated}'  ({mt_ms} ms)")

            send_json(self, {
                "translation": translated,
                "tgt_lang":    tgt_lang,
                "metrics":     {"mt_time_ms": mt_ms}
            })

        # ── /tts  ── slow OmniVoice synthesis, called after /translate ─────────
        elif self.path == '/tts':
            text     = req.get('text', '').strip()
            tgt_lang = req.get('tgt_lang', 'ja')   # language of the text to speak

            if not text:
                send_json(self, {"audio_b64": None, "metrics": {"tts_time_ms": 0}})
                return

            # Pick voice prompt based on target language
            if tgt_lang == 'ja':
                instruct = "female, young adult, moderate pitch, japanese accent"
            else:
                instruct = "female, young adult, moderate pitch, american accent"

            print(f"\n[TTS] text='{text}'  lang={tgt_lang}  prompt='{instruct}'")
            t0 = time.time()

            try:
                with torch.inference_mode():
                    audio_list = tts_model.generate(
                        text=text,
                        instruct=instruct,
                        num_step=16,
                        speed=1.0
                    )

                if not audio_list:
                    raise ValueError("OmniVoice returned empty list")

                audio_b64 = audio_list_to_wav_b64(audio_list, sample_rate=24000)
                tts_ms    = int((time.time() - t0) * 1000)
                print(f"[TTS] Done in {tts_ms} ms")

                send_json(self, {
                    "audio_b64": audio_b64,
                    "metrics":   {"tts_time_ms": tts_ms}
                })

            except Exception as e:
                print(f"[TTS] ERROR: {e}")
                send_json(self, {
                    "audio_b64": None,
                    "error":     str(e),
                    "metrics":   {"tts_time_ms": -1}
                })

        else:
            send_json(self, {"error": "Not found"}, status=404)


# ── Entry point ────────────────────────────────────────────────────────────────
def run(port=8000):
    httpd = HTTPServer(('', port), HybridHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

if __name__ == '__main__':
    run()
