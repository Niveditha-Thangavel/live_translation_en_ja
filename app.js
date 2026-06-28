import { pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Set up huggingface environment settings if needed
env.allowLocalModels = false;

// DOM Elements
const gpuBadge = document.getElementById('gpu-badge');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const btnListen = document.getElementById('btn-listen');
const btnEnJa = document.getElementById('btn-en-ja');
const btnJaEn = document.getElementById('btn-ja-en');
const recordingPulse = document.getElementById('recording-pulse');
const canvas = document.getElementById('waveform-canvas');
const ctx = canvas.getContext('2d');
const placeholderText = document.getElementById('visualizer-placeholder');

const textSrc = document.getElementById('text-src');
const textTgt = document.getElementById('text-tgt');
const srcLangLabel = document.getElementById('src-lang-label');
const tgtLangLabel = document.getElementById('tgt-lang-label');

const metricStt = document.getElementById('metric-stt');
const metricMt = document.getElementById('metric-mt');
const metricTts = document.getElementById('metric-tts');

const progressWhisperVal = document.getElementById('progress-whisper-val');
const progressWhisperBar = document.getElementById('progress-whisper-bar');
const progressSmall100Val = document.getElementById('progress-small100-val');
const progressSmall100Bar = document.getElementById('progress-small100-bar');

// App Variables
let currentState = 'UNINITIALIZED'; // UNINITIALIZED, LOADING, IDLE, SPEAKING, TRANSLATING, SYNTHESIZING
let srcLang = 'en'; // 'en' or 'ja'
let tgtLang = 'ja'; // 'ja' or 'en'
let hasWebGPU = false;
let audioContext = null;      // 16kHz context for mic capture
let playbackContext = null;   // 24kHz context for TTS playback
let audioWorkletNode = null;
let micStream = null;

// AI Model Pipelines
let sttPipeline = null;
let mtTokenizer = null;
let mtModel = null;

// Check for WebGPU support
async function checkWebGPU() {
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        hasWebGPU = true;
        gpuBadge.textContent = 'WebGPU Accelerated';
        gpuBadge.classList.add('green');
        console.log('WebGPU is available and accelerated!');
        return;
      }
    } catch (e) {
      console.warn('WebGPU request adapter failed:', e);
    }
  }
  gpuBadge.textContent = 'CPU (WASM Mode)';
  console.log('Falling back to WASM execution.');
}

// State Machine Guard
function updateState(newState) {
  currentState = newState;
  console.log(`[State Transition]: ${newState}`);
  
  switch(newState) {
    case 'UNINITIALIZED':
      statusText.textContent = 'Click to Initialize';
      statusDot.className = 'dot';
      btnListen.disabled = false;
      btnListen.querySelector('.btn-text').textContent = 'Initialize Models';
      recordingPulse.classList.add('hide');
      break;
      
    case 'LOADING':
      statusText.textContent = 'Downloading Models...';
      statusDot.className = 'dot';
      btnListen.disabled = true;
      btnListen.querySelector('.btn-text').textContent = 'Loading...';
      recordingPulse.classList.add('hide');
      break;
      
    case 'IDLE':
      statusText.textContent = 'Ready (Idle)';
      statusDot.className = 'dot green';
      btnListen.disabled = false;
      btnListen.classList.remove('recording');
      btnListen.querySelector('.btn-text').textContent = 'Start Listening';
      recordingPulse.classList.add('hide');
      break;
      
    case 'SPEAKING':
      statusText.textContent = 'Listening...';
      statusDot.className = 'dot green';
      btnListen.disabled = false;
      btnListen.classList.add('recording');
      btnListen.querySelector('.btn-text').textContent = 'Stop Listening';
      recordingPulse.classList.remove('hide');
      break;
      
    case 'TRANSLATING':
      statusText.textContent = 'Processing Speech...';
      statusDot.className = 'dot';
      recordingPulse.classList.add('hide');
      break;
      
    case 'SYNTHESIZING':
      statusText.textContent = 'Speaking...';
      statusDot.className = 'dot green';
      recordingPulse.classList.add('hide');
      break;
  }
}

// Initialize the Pipelines
async function initializePipelines() {
  updateState('LOADING');
  
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  const progressCallbacks = {
    whisper: (data) => {
      if (data.status === 'progress') {
        const percent = Math.round(data.progress);
        progressWhisperVal.textContent = `${percent}%`;
        progressWhisperBar.style.width = `${percent}%`;
      } else if (data.status === 'ready') {
        progressWhisperVal.textContent = 'Loaded';
        progressWhisperBar.style.width = '100%';
      }
    }
  };

  try {
    // 1. Load Whisper STT (in browser)
    sttPipeline = await pipeline(
      'automatic-speech-recognition', 
      'Xenova/whisper-tiny', 
      { 
        device, 
        quantized: true,
        progress_callback: progressCallbacks.whisper 
      }
    );

    // Update status for small100 to show it is offloaded to CPU
    progressSmall100Val.textContent = 'Offloaded to CPU';
    progressSmall100Bar.style.width = '100%';

    updateState('IDLE');
  } catch (error) {
    console.error('Pipeline Initialization Failed:', error);
    statusText.textContent = 'Initialization Failed';
    statusDot.className = 'dot red';
  }
}

/**
 * Decode a Base64-encoded WAV string and play it through the Web Audio API.
 * Uses a dedicated 24kHz playback context, separate from the 16kHz mic capture context.
 * Returns a Promise that resolves when playback finishes.
 */
async function playAudioB64(b64String) {
  if (!b64String) {
    console.warn('[TTS] No audio data received from server.');
    return;
  }

  updateState('SYNTHESIZING');
  console.log(`[TTS] Received ${b64String.length} Base64 chars, decoding...`);

  // Use a dedicated playback context at 24kHz (OmniVoice sample rate)
  // Never reuse the 16kHz mic context — sample rate mismatch causes silent output
  if (!playbackContext || playbackContext.state === 'closed') {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
  }
  if (playbackContext.state === 'suspended') {
    await playbackContext.resume();
  }

  // Base64 → Uint8Array → ArrayBuffer
  const binaryStr = atob(b64String);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  const arrayBuf = bytes.buffer;

  return new Promise((resolve, reject) => {
    // Use callback form for maximum browser compatibility
    playbackContext.decodeAudioData(
      arrayBuf,
      (audioBuffer) => {
        console.log(`[TTS] Decoded OK — duration: ${audioBuffer.duration.toFixed(2)}s, channels: ${audioBuffer.numberOfChannels}`);
        const source = playbackContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackContext.destination);
        source.onended = () => {
          console.log('[TTS] Playback finished.');
          resolve();
        };
        source.start(0);
      },
      (err) => {
        console.error('[TTS] decodeAudioData failed:', err);
        reject(err);
      }
    );
  });
}

// Main processing loop: speech -> text -> translation -> tts
async function processSpeechAudio(audioPCM) {
  // If we are already translating or synthesizing, ignore microphone input
  if (currentState === 'TRANSLATING' || currentState === 'SYNTHESIZING') return;
  
  updateState('TRANSLATING');
  
  textSrc.textContent = 'Transcribing...';
  textSrc.classList.remove('empty');
  textTgt.textContent = 'Translating...';
  textTgt.classList.remove('empty');
  
  let sttTime = 0;
  let mtTime = 0;

  try {
    // A. Speech to Text (Whisper in Browser)
    const t0 = performance.now();
    const sttResult = await sttPipeline(audioPCM, {
      chunk_length_s: 30,
      stride_length_s: 5,
      language: srcLang === 'ja' ? 'japanese' : 'english',
      task: 'transcribe'
    });
    const transcribedText = sttResult.text.trim();
    sttTime = Math.round(performance.now() - t0);
    metricStt.textContent = `${sttTime} ms`;
    
    textSrc.textContent = transcribedText || '(No clear speech detected)';

    // --- Hallucination filter ---
    // Whisper outputs these special tokens for silence/noise - never translate them
    const WHISPER_HALLUCINATIONS = /^\s*(\[BLANK_AUDIO\]|\[blank_audio\]|\[Silence\]|\[silence\]|\[SILENCE\]|\[Music\]|\[music\]|\[Noise\]|\[noise\]|\.{1,5}|\s*)\s*$/i;
    if (!transcribedText || WHISPER_HALLUCINATIONS.test(transcribedText)) {
      console.log(`[STT] Filtered hallucination: '${transcribedText}'`);
      updateState('SPEAKING');
      textSrc.textContent = '(No clear speech detected)';
      textSrc.classList.add('empty');
      return;
    }

    // ── B. Machine Translation (fast, ~200ms) ──────────────────────────────
    console.log(`[Client] → /translate  '${transcribedText}'`);
    const mtRes = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: transcribedText, src_lang: srcLang })
    });
    if (!mtRes.ok) throw new Error(`/translate returned ${mtRes.status}`);

    const mtData = await mtRes.json();
    const translatedText = mtData.translation;
    const tgtLangReturned = mtData.tgt_lang || (srcLang === 'en' ? 'ja' : 'en');

    // Show text immediately — user doesn't have to wait for TTS
    textTgt.textContent = translatedText;
    textTgt.classList.remove('empty');
    mtTime = mtData.metrics.mt_time_ms;
    metricMt.textContent = `${mtTime} ms`;
    metricTts.textContent = 'Synthesizing…';

    // Return to SPEAKING so mic is live again while TTS runs in background
    updateState('SPEAKING');

    // ── C. OmniVoice TTS (slow, ~60–90s CPU — runs in background) ──────────
    if (translatedText) {
      console.log(`[Client] → /tts  '${translatedText}'  (lang=${tgtLangReturned})`);
      try {
        const ttsRes = await fetch('/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: translatedText, tgt_lang: tgtLangReturned })
        });

        if (ttsRes.ok) {
          const ttsData = await ttsRes.json();
          metricTts.textContent = `${ttsData.metrics.tts_time_ms} ms (CPU)`;

          if (ttsData.audio_b64) {
            await playAudioB64(ttsData.audio_b64);
          } else {
            console.warn('[TTS] Server returned no audio:', ttsData.error || 'unknown reason');
            metricTts.textContent = 'TTS failed';
          }
        } else {
          console.warn(`[TTS] /tts returned status ${ttsRes.status}`);
          metricTts.textContent = 'TTS error';
        }
      } catch (ttsErr) {
        console.error('[TTS] fetch error:', ttsErr);
        metricTts.textContent = 'TTS error';
      } finally {
        // Make sure we're back in SPEAKING after playback
        updateState('SPEAKING');
      }
    }

  } catch (error) {
    console.error('Speech translation pipeline failed:', error);
    textTgt.textContent = 'Error processing translation.';
    updateState('SPEAKING');
  }
}

// Canvas Waveform visualizer
let vizBuffer = [];
function drawWaveform() {
  requestAnimationFrame(drawWaveform);
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (currentState !== 'SPEAKING') {
    placeholderText.style.display = 'flex';
    return;
  }
  
  placeholderText.style.display = 'none';
  
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#3b82f6';
  ctx.beginPath();
  
  const sliceWidth = canvas.width / vizBuffer.length;
  let x = 0;
  
  for (let i = 0; i < vizBuffer.length; i++) {
    const v = vizBuffer[i] * 1.5; // Amplify slightly for visualization
    const y = (v * canvas.height / 2) + (canvas.height / 2);
    
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    
    x += sliceWidth;
  }
  
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

// Adjust canvas resolution for high-DPI displays
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
}

// Setup Microphone & AudioWorklet capture node
async function startAudioCapture() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000 // Whisper expects 16kHz audio input
    });
    
    await audioContext.audioWorklet.addModule('audio-worklet.js');
  }
  
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true
    }
  });

  const source = audioContext.createMediaStreamSource(micStream);
  audioWorkletNode = new AudioWorkletNode(audioContext, 'vad-processor');

  // Handle messages from VAD processor
  audioWorkletNode.port.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'audio-chunk-viz') {
      vizBuffer = msg.samples;
    } else if (msg.type === 'speech-start') {
      console.log('VAD: Speech started');
      textSrc.textContent = 'Listening...';
    } else if (msg.type === 'speech-end') {
      console.log('VAD: Speech completed, processing chunk size:', msg.audioPCM.length);
      processSpeechAudio(msg.audioPCM);
    }
  };

  source.connect(audioWorkletNode);
  audioWorkletNode.connect(audioContext.destination);
  
  updateState('SPEAKING');
}

// Stop Audio Capture
function stopAudioCapture() {
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
  }
  if (audioWorkletNode) {
    audioWorkletNode.disconnect();
    audioWorkletNode = null;
  }
  updateState('IDLE');
}

// Click listener to toggle listening stream
btnListen.addEventListener('click', async () => {
  if (currentState === 'UNINITIALIZED') {
    await initializePipelines();
  } else if (currentState === 'IDLE') {
    try {
      await startAudioCapture();
    } catch (err) {
      console.error('Failed to get microphone:', err);
      alert('Microphone access is required to use this application.');
    }
  } else if (currentState === 'SPEAKING' || currentState === 'TRANSLATING' || currentState === 'SYNTHESIZING') {
    stopAudioCapture();
  }
});

// Direction Toggles
btnEnJa.addEventListener('click', () => {
  if (srcLang === 'en') return;
  srcLang = 'en';
  tgtLang = 'ja';
  btnEnJa.classList.add('active');
  btnJaEn.classList.remove('active');
  srcLangLabel.textContent = 'English';
  tgtLangLabel.textContent = 'Japanese';
  textSrc.textContent = 'Waiting for voice input...';
  textSrc.classList.add('empty');
  textTgt.textContent = 'Waiting for translation...';
  textTgt.classList.add('empty');
});

btnJaEn.addEventListener('click', () => {
  if (srcLang === 'ja') return;
  srcLang = 'ja';
  tgtLang = 'en';
  btnJaEn.classList.add('active');
  btnEnJa.classList.remove('active');
  srcLangLabel.textContent = 'Japanese';
  tgtLangLabel.textContent = 'English';
  textSrc.textContent = 'Waiting for voice input...';
  textSrc.classList.add('empty');
  textTgt.textContent = 'Waiting for translation...';
  textTgt.classList.add('empty');
});

// Init
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
checkWebGPU().then(() => {
  updateState('UNINITIALIZED');
});
drawWaveform();
