/**
 * ShieldVoice (SIH26104) - Live Interactive Testing Studio Frontend Logic
 * Implements sample-accurate 16kHz downsampling and PCM encoding to eliminate browser sample-rate and codec distortion.
 */

// State
let currentTab = 'record';
let audioContext = null;
let micStream = null;
let micSource = null;
let scriptProcessor = null;
let analyser = null;
let audioSamples = [];
let isRecording = false;

let recordingInterval = null;
let recordingSeconds = 0;
let recordedBlob = null;
let selectedUploadFile = null;
let lastAnalysisResult = null;

document.addEventListener('DOMContentLoaded', () => {
    initVisualizerCanvas();
    loadSoundboard();
    setupDropzone();
});

// TAB NAVIGATION
function switchTab(tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = document.getElementById(`tab-${tabId}`);
    const activeView = document.getElementById(`view-${tabId}`);

    if (activeBtn) activeBtn.classList.add('active');
    if (activeView) activeView.classList.add('active');

    if (tabId !== 'record' && isRecording) {
        toggleRecording();
    }
}

// VISUALIZER CANVAS
function initVisualizerCanvas() {
    const canvas = document.getElementById('micCanvas');
    const ctx = canvas.getContext('2d');
    
    let phase = 0;
    function drawIdle() {
        if (isRecording) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
        ctx.beginPath();
        
        const sliceWidth = canvas.width / 50;
        let x = 0;
        
        for (let i = 0; i < 50; i++) {
            const v = Math.sin(phase + i * 0.2) * 8;
            const y = (canvas.height / 2) + v;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }
        ctx.stroke();
        phase += 0.04;
        requestAnimationFrame(drawIdle);
    }
    drawIdle();
}

// SAMPLE-ACCURATE DOWNSAMPLER TO 16kHz
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
    if (inputSampleRate === outputSampleRate) {
        return buffer;
    }
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = count > 0 ? (accum / count) : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

// MICROPHONE RECORDING IN PURE 16kHz PCM
async function toggleRecording() {
    const btnRecord = document.getElementById('btnRecord');
    const btnRecordText = document.getElementById('btnRecordText');
    const micStatus = document.getElementById('micStatus');
    const btnAnalyze = document.getElementById('btnAnalyzeRec');
    const playback = document.getElementById('recordedAudioPlayback');

    if (!isRecording) {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: true
                }
            });

            audioSamples = [];
            
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioCtx();
            
            micSource = audioContext.createMediaStreamSource(micStream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                audioSamples.push(new Float32Array(inputData));
            };

            micSource.connect(analyser);
            analyser.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            isRecording = true;

            btnRecord.classList.add('recording');
            btnRecordText.innerText = 'Stop Recording';
            micStatus.innerText = '● RECORDING LIVE (Please speak naturally for 3-5 sec)...';
            micStatus.style.color = '#ef4444';
            btnAnalyze.disabled = true;

            recordingSeconds = 0;
            updateTimerDisplay();
            recordingInterval = setInterval(() => {
                recordingSeconds++;
                updateTimerDisplay();
                if (recordingSeconds >= 8) toggleRecording();
            }, 1000);

            drawLiveWaveform();
        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Could not access microphone. Please check browser permissions.");
        }
    } else {
        isRecording = false;
        clearInterval(recordingInterval);

        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor.onaudioprocess = null;
        }
        if (micSource) micSource.disconnect();
        if (micStream) micStream.getTracks().forEach(track => track.stop());

        // Merge raw chunks
        let totalSamplesCount = 0;
        for (let chunk of audioSamples) totalSamplesCount += chunk.length;
        
        let mergedSamples = new Float32Array(totalSamplesCount);
        let offset = 0;
        for (let chunk of audioSamples) {
            mergedSamples.set(chunk, offset);
            offset += chunk.length;
        }

        // Accurate Resampling to 16,000 Hz
        const nativeSampleRate = audioContext ? audioContext.sampleRate : 44100;
        const resampled16k = downsampleBuffer(mergedSamples, nativeSampleRate, 16000);

        // Encode to standard 16kHz 16-bit PCM WAV
        recordedBlob = encodeWAV(resampled16k, 16000);

        playback.src = URL.createObjectURL(recordedBlob);
        playback.classList.remove('hidden');
        btnAnalyze.disabled = false;

        btnRecord.classList.remove('recording');
        btnRecordText.innerText = 'Record Again';
        micStatus.innerText = `✓ 16kHz Studio Audio Ready (${(resampled16k.length / 16000).toFixed(1)}s)`;
        micStatus.style.color = '#10b981';
    }
}

function updateTimerDisplay() {
    const mins = Math.floor(recordingSeconds / 60).toString().padStart(2, '0');
    const secs = (recordingSeconds % 60).toString().padStart(2, '0');
    document.getElementById('recTimer').innerText = `${mins}:${secs}`;
}

// RIFF 16-Bit Linear PCM WAV Encoder
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let byteOffset = 44;
    for (let i = 0; i < samples.length; i++, byteOffset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(byteOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function drawLiveWaveform() {
    if (!isRecording || !analyser) return;

    const canvas = document.getElementById('micCanvas');
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function render() {
        if (!isRecording) return;
        requestAnimationFrame(render);
        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = '#060911';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#00f0ff';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00f0ff';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * (canvas.height / 2);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    render();
}

function analyzeRecordedClip() {
    if (!recordedBlob) return;
    const file = new File([recordedBlob], `live_mic_capture_${Date.now()}.wav`, { type: 'audio/wav' });
    sendAudioForAnalysis(file);
}

// FILE UPLOAD & DROPZONE
function setupDropzone() {
    const dropzone = document.getElementById('dropZone');
    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleAudioFile(files[0]);
        }
    });
}

function handleFileSelected(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleAudioFile(files[0]);
    }
}

function handleAudioFile(file) {
    selectedUploadFile = file;
    document.getElementById('previewFileName').innerText = file.name;
    document.getElementById('previewFileSize').innerText = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    
    const player = document.getElementById('fileAudioPlayer');
    player.src = URL.createObjectURL(file);
    document.getElementById('filePreviewCard').classList.remove('hidden');
}

function analyzeUploadedFile() {
    if (!selectedUploadFile) return;
    sendAudioForAnalysis(selectedUploadFile);
}

// SOUNDBOARD PRESETS
async function loadSoundboard() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        if (data.status === 'success') {
            renderSoundboardCards(data.samples);
        }
    } catch (err) {
        console.error("Failed to load soundboard samples:", err);
    }
}

function renderSoundboardCards(samples) {
    const container = document.getElementById('soundboardGrid');
    container.innerHTML = '';

    samples.forEach(sample => {
        const isHuman = sample.category === 'human';
        const card = document.createElement('div');
        card.className = `sound-card category-${sample.category}`;
        card.innerHTML = `
            <div>
                <div class="sound-card-header">
                    <span class="sound-tag ${isHuman ? 'human' : 'ai'}">
                        ${isHuman ? '🟢 GENUINE HUMAN' : '🔴 AI DEEPFAKE'}
                    </span>
                </div>
                <h4>${sample.name}</h4>
                <p>${sample.description}</p>
            </div>
            <div class="sound-card-actions">
                <button class="btn-sound-play" onclick="playSoundboardSample('/api/samples/${sample.id}')">
                    ▶ Play
                </button>
                <button class="btn-sound-test" onclick="testSoundboardSample('${sample.id}', '${sample.name}')">
                    ⚡ Instant Test
                </button>
            </div>
        `;
        container.appendChild(card);
    });
}

let activeAudio = null;
function playSoundboardSample(url) {
    if (activeAudio) {
        activeAudio.pause();
    }
    activeAudio = new Audio(url);
    activeAudio.play();
}

async function testSoundboardSample(sampleId, sampleName) {
    try {
        setAnalyzingState(true);
        const res = await fetch(`/api/samples/${sampleId}`);
        const blob = await res.blob();
        const file = new File([blob], sampleId, { type: 'audio/wav' });
        await sendAudioForAnalysis(file);
    } catch (err) {
        console.error("Soundboard test error:", err);
        setAnalyzingState(false);
    }
}

// INFERENCE EXECUTION
async function sendAudioForAnalysis(file) {
    setAnalyzingState(true);
    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/api/analyze-audio', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
            lastAnalysisResult = data;
            renderResults(data);
        } else {
            alert(`Analysis Error: ${data.message}`);
        }
    } catch (err) {
        console.error("Inference request failed:", err);
        alert("Failed to communicate with ShieldVoice inference engine.");
    } finally {
        setAnalyzingState(false);
    }
}

function setAnalyzingState(isAnalyzing) {
    const badge = document.getElementById('engineStatus');
    if (isAnalyzing) {
        badge.innerText = '● INFERENCE COMPUTING...';
        badge.style.color = '#00f0ff';
    } else {
        badge.innerText = 'Engine Ready';
        badge.style.color = '#9ca3af';
    }
}

// TELEMETRY & RESULTS RENDERING
function renderResults(data) {
    const risk = data.spoof_risk_percent;
    const humanConf = data.human_confidence_percent;
    const isSpoof = data.prediction === 'SPOOF / DEEPFAKE';

    const gaugeFill = document.getElementById('gaugeFill');
    const riskDisplay = document.getElementById('riskScoreDisplay');
    
    const circumference = 502.65;
    const offset = circumference - (circumference * (risk / 100));
    gaugeFill.style.strokeDashoffset = offset;
    riskDisplay.innerText = `${risk.toFixed(1)}%`;

    if (risk < 40) {
        gaugeFill.style.stroke = '#10b981';
        riskDisplay.style.color = '#10b981';
    } else if (risk < 70) {
        gaugeFill.style.stroke = '#f59e0b';
        riskDisplay.style.color = '#f59e0b';
    } else {
        gaugeFill.style.stroke = '#ef4444';
        riskDisplay.style.color = '#ef4444';
    }

    const verdictBadge = document.getElementById('verdictBadge');
    const threatPill = document.getElementById('threatPill');

    verdictBadge.innerText = data.prediction;
    verdictBadge.className = `verdict-badge ${isSpoof ? 'badge-spoof' : 'badge-human'}`;
    threatPill.innerText = `THREAT LEVEL: ${data.threat_level}`;

    document.getElementById('valHumanConf').innerText = `${humanConf.toFixed(1)}%`;
    document.getElementById('valSpoofRisk').innerText = `${risk.toFixed(1)}%`;
    document.getElementById('valLatency').innerText = `${data.inference_latency_ms} ms`;
    document.getElementById('valSampleRate').innerText = `${data.diagnostics.sample_rate_hz || 16000} Hz`;

    const diag = data.diagnostics;
    document.getElementById('diagVocoder').innerText = `${diag.vocoder_artifact_density}%`;
    document.getElementById('barVocoder').style.width = `${diag.vocoder_artifact_density}%`;

    document.getElementById('diagProsody').innerText = `${diag.micro_prosody_organic_score}%`;
    document.getElementById('barProsody').style.width = `${diag.micro_prosody_organic_score}%`;

    document.getElementById('diagPhase').innerText = `${diag.phase_continuity_index}%`;
    document.getElementById('barPhase').style.width = `${diag.phase_continuity_index}%`;

    const fluxBadge = document.getElementById('diagFlux');
    fluxBadge.innerText = diag.spectral_flux_stability;
    fluxBadge.style.color = isSpoof ? '#ef4444' : '#10b981';
}

function exportJsonReport() {
    if (!lastAnalysisResult) {
        alert("Please analyze an audio clip first before exporting telemetry.");
        return;
    }
    const reportData = {
        system: "ShieldVoice (SIH26104)",
        timestamp: new Date().toISOString(),
        model_backbone: "Wav2Vec2 + AASIST Graph Attention Network",
        telemetry: lastAnalysisResult
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ShieldVoice_Threat_Telemetry_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
