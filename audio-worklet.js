class VADProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // VAD Parameters
    this.sampleRate = 16000; // Whisper expects 16kHz
    this.energyThreshold = 0.003;  // RMS energy threshold for voice activity

    // Require 700ms of silence before cutting a segment (was 250ms).
    // Shorter values fire too eagerly on breath/background noise.
    this.silenceTimeoutFrames = Math.round(0.7 * this.sampleRate / 128);

    // Minimum speech duration to bother sending (0.5 s = 8000 samples).
    // Prevents hallucinations from near-empty buffers.
    this.MIN_SPEECH_SAMPLES = 8000;

    // Buffers and states
    this.isSpeechActive = false;
    this.silenceCounter = 0;
    this.audioBuffer = []; // Accumulated float32 PCM blocks
    this.totalSamples = 0; // Running count of accumulated samples
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    
    // We capture channel 0 (mono)
    const channelData = input[0];
    if (!channelData) return true;

    // Send visualizer data to main thread
    this.port.postMessage({
      type: 'audio-chunk-viz',
      samples: channelData
    });

    // Calculate RMS energy of this 128-sample block
    let sum = 0;
    for (let i = 0; i < channelData.length; i++) {
      sum += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sum / channelData.length);

    // VAD decision
    if (rms > this.energyThreshold) {
      // Speech detected
      if (!this.isSpeechActive) {
        this.isSpeechActive = true;
        this.port.postMessage({ type: 'speech-start' });
      }
      this.silenceCounter = 0;
    } else {
      // Silence detected
      if (this.isSpeechActive) {
        this.silenceCounter++;
        if (this.silenceCounter >= this.silenceTimeoutFrames) {
          // Silence threshold reached — end of utterance
          this.isSpeechActive = false;
          this.silenceCounter = 0;
          
          // Only fire if we accumulated enough real speech samples
          if (this.audioBuffer.length > 0 && this.totalSamples >= this.MIN_SPEECH_SAMPLES) {
            const flatBuffer = this.flattenBuffer(this.audioBuffer);
            this.port.postMessage({
              type: 'speech-end',
              audioPCM: flatBuffer
            });
          } else {
            console.log(`[VAD] Discarded short chunk (${this.totalSamples} samples < ${this.MIN_SPEECH_SAMPLES} minimum)`);
          }
          this.audioBuffer = [];
          this.totalSamples = 0;
        }
      }
    }

    // If active speech, accumulate samples
    if (this.isSpeechActive) {
      this.audioBuffer.push(new Float32Array(channelData));
      this.totalSamples += channelData.length;
    }

    return true;
  }

  flattenBuffer(chunks) {
    let totalLength = 0;
    for (let i = 0; i < chunks.length; i++) {
      totalLength += chunks[i].length;
    }
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      result.set(chunks[i], offset);
      offset += chunks[i].length;
    }
    return result;
  }
}

registerProcessor('vad-processor', VADProcessor);
