/**
 * Miss Riverwood - Real-time WebRTC Voice Assistant
 * Handles audio streaming, STT, LLM, TTS pipeline with visual feedback
 */

class VoiceAssistant {
    constructor() {
        // WebSocket connection
        this.ws = null;
        this.isConnected = false;
        this.isRecording = false;
        
        // Audio handling
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioContext = null;
        this.audioPlayer = document.getElementById('audioPlayer');
        
        // UI elements
        this.startCallBtn = document.getElementById('startCallBtn');
        this.endCallBtn = document.getElementById('endCallBtn');
        this.statusBadge = document.getElementById('statusBadge');
        this.agentAvatar = document.getElementById('agentAvatar');
        this.messagesContainer = document.getElementById('messages');
        this.conversationContainer = document.getElementById('conversationContainer');
        
        // Pipeline steps
        this.sttStep = document.getElementById('sttStep');
        this.llmStep = document.getElementById('llmStep');
        this.ttsStep = document.getElementById('ttsStep');
        this.memoryStep = document.getElementById('memoryStep');
        
        // Stats
        this.startTime = null;
        this.latencyDisplay = document.getElementById('latencyDisplay');
        
        // Bind event listeners
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.startCallBtn.addEventListener('click', () => this.startCall());
        this.endCallBtn.addEventListener('click', () => this.endCall());
    }
    
    async startCall() {
        try {
            console.log('🎤 Starting call...');
            this.updateStatus('connecting', 'Connecting...');
            this.startCallBtn.disabled = true;
            
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                } 
            });
            
            console.log('✅ Microphone access granted');
            
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Setup MediaRecorder
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            };
            
            try {
                this.mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                console.warn('⚠️ webm not supported, trying default format');
                this.mediaRecorder = new MediaRecorder(stream);
            }
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('🎤 Recording stopped, processing audio...');
                this.processAudioChunks();
            };
            
            // Connect to WebSocket
            await this.connectWebSocket();
            
            // Update UI
            this.updateStatus('connected', 'Connected');
            this.startCallBtn.style.display = 'none';
            this.endCallBtn.style.display = 'inline-flex';
            this.clearEmptyState();
            
            console.log('✅ Call started successfully');
            
        } catch (error) {
            console.error('❌ Error starting call:', error);
            this.updateStatus('error', 'Error');
            this.startCallBtn.disabled = false;
            this.showError('Could not start call. Please check microphone permissions.');
        }
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/voice-stream`;
            
            console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.isConnected = true;
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.isConnected = false;
                this.updateStatus('disconnected', 'Disconnected');
            };
            
            // Timeout after 5 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            console.log('📨 Received message:', message.type);
            
            switch (message.type) {
                case 'greeting':
                    this.handleGreeting(message);
                    break;
                case 'response':
                    this.handleResponse(message);
                    break;
                case 'error':
                    this.handleError(message);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
        }
    }
    
    async handleGreeting(message) {
        console.log('👋 Greeting received:', message.text);
        
        // Add greeting to conversation
        this.addMessage('assistant', message.text);
        
        // Play greeting audio
        if (message.audio) {
            await this.playAudioHex(message.audio);
        }
        
        // Start listening for user input after greeting
        this.startListening();
    }
    
    async handleResponse(message) {
        console.log('💬 Response received:', message.text);
        
        // Calculate latency
        if (this.startTime) {
            const latency = Date.now() - this.startTime;
            this.latencyDisplay.textContent = `Latency: ${latency}ms`;
            this.startTime = null;
        }
        
        // Reset pipeline
        this.resetPipeline();
        
        // Add user message (transcript)
        this.addMessage('user', message.transcript);
        
        // Add assistant response
        this.addMessage('assistant', message.text);
        
        // Play response audio
        if (message.audio) {
            await this.playAudioHex(message.audio);
        }
        
        // Start listening again
        setTimeout(() => {
            this.startListening();
        }, 500);
    }
    
    handleError(message) {
        console.error('❌ Error from server:', message.message);
        this.showError(message.message);
        this.resetPipeline();
        
        // Retry listening after error
        setTimeout(() => {
            this.startListening();
        }, 1000);
    }
    
    startListening() {
        if (!this.isConnected || this.isRecording) return;
        
        console.log('👂 Starting to listen...');
        this.isRecording = true;
        this.audioChunks = [];
        this.updateStatus('listening', 'Listening...');
        
        // Activate STT pipeline step
        this.activatePipelineStep('stt');
        
        // Start recording for 5 seconds (adjust as needed)
        this.mediaRecorder.start();
        
        // Auto-stop after 5 seconds
        setTimeout(() => {
            if (this.isRecording && this.mediaRecorder.state === 'recording') {
                console.log('⏱️ Recording timeout, stopping...');
                this.stopRecording();
            }
        }, 5000);
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        console.log('⏹️ Stopping recording...');
        this.isRecording = false;
        this.updateStatus('processing', 'Processing...');
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }
    
    async processAudioChunks() {
        if (this.audioChunks.length === 0) {
            console.warn('⚠️ No audio chunks to process');
            this.startListening();
            return;
        }
        
        console.log(`🎵 Processing ${this.audioChunks.length} audio chunks`);
        
        // Mark start time for latency measurement
        this.startTime = Date.now();
        
        // Activate LLM pipeline step
        this.activatePipelineStep('llm');
        
        try {
            // Combine audio chunks into single blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            console.log(`📦 Audio blob size: ${audioBlob.size} bytes`);
            
            // Convert blob to hex string for WebSocket transmission
            const arrayBuffer = await audioBlob.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);
            const hexString = Array.from(uint8Array)
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
            
            console.log(`📤 Sending audio to server (${hexString.length} hex chars)`);
            
            // Send to server
            this.ws.send(JSON.stringify({
                type: 'audio',
                audio: hexString
            }));
            
            // Activate remaining pipeline steps
            setTimeout(() => this.activatePipelineStep('tts'), 500);
            setTimeout(() => this.activatePipelineStep('memory'), 1000);
            
        } catch (error) {
            console.error('❌ Error processing audio:', error);
            this.showError('Failed to process audio');
            this.resetPipeline();
            this.startListening();
        }
    }
    
    async playAudioHex(hexString) {
        try {
            console.log(`🔊 Playing audio (${hexString.length} hex chars)`);
            
            // Show speaking animation
            this.agentAvatar.classList.add('speaking');
            
            // Convert hex to byte array
            const bytes = new Uint8Array(hexString.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
            
            // Create blob and URL
            const blob = new Blob([bytes], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            // Play audio
            return new Promise((resolve, reject) => {
                this.audioPlayer.src = url;
                this.audioPlayer.onended = () => {
                    console.log('✅ Audio playback finished');
                    this.agentAvatar.classList.remove('speaking');
                    URL.revokeObjectURL(url);
                    resolve();
                };
                this.audioPlayer.onerror = (error) => {
                    console.error('❌ Audio playback error:', error);
                    this.agentAvatar.classList.remove('speaking');
                    URL.revokeObjectURL(url);
                    reject(error);
                };
                this.audioPlayer.play();
            });
        } catch (error) {
            console.error('❌ Error playing audio:', error);
            this.agentAvatar.classList.remove('speaking');
        }
    }
    
    endCall() {
        console.log('📞 Ending call...');
        
        // Stop recording
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        // Close WebSocket
        if (this.ws) {
            this.ws.send(JSON.stringify({ type: 'end' }));
            this.ws.close();
        }
        
        // Reset UI
        this.updateStatus('ready', 'Ready');
        this.startCallBtn.style.display = 'inline-flex';
        this.startCallBtn.disabled = false;
        this.endCallBtn.style.display = 'none';
        this.agentAvatar.classList.remove('speaking');
        this.resetPipeline();
        
        console.log('✅ Call ended');
    }
    
    addMessage(sender, text) {
        // Remove empty state if present
        const emptyState = this.messagesContainer.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        const senderName = sender === 'user' ? 'You' : 'Miss Riverwood';
        const time = new Date().toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-sender">${senderName}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-text">${text}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    clearEmptyState() {
        const emptyState = this.messagesContainer.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
    }
    
    showError(message) {
        this.addMessage('assistant', `⚠️ ${message}`);
    }
    
    updateStatus(state, text) {
        const statusDot = this.statusBadge.querySelector('.status-dot');
        const statusText = this.statusBadge.querySelector('span');
        
        statusText.textContent = text;
        
        // Update badge styling based on state
        this.statusBadge.className = 'status-badge';
        
        switch (state) {
            case 'connected':
            case 'listening':
                this.statusBadge.style.background = 'rgba(74, 222, 128, 0.1)';
                this.statusBadge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
                this.statusBadge.style.color = '#4ade80';
                break;
            case 'processing':
                this.statusBadge.style.background = 'rgba(251, 191, 36, 0.1)';
                this.statusBadge.style.borderColor = 'rgba(251, 191, 36, 0.3)';
                this.statusBadge.style.color = '#fbbf24';
                break;
            case 'connecting':
                this.statusBadge.style.background = 'rgba(102, 126, 234, 0.1)';
                this.statusBadge.style.borderColor = 'rgba(102, 126, 234, 0.3)';
                this.statusBadge.style.color = '#667eea';
                break;
            case 'error':
            case 'disconnected':
                this.statusBadge.style.background = 'rgba(244, 63, 94, 0.1)';
                this.statusBadge.style.borderColor = 'rgba(244, 63, 94, 0.3)';
                this.statusBadge.style.color = '#f43f5e';
                break;
            default:
                this.statusBadge.style.background = 'rgba(163, 163, 194, 0.1)';
                this.statusBadge.style.borderColor = 'rgba(163, 163, 194, 0.3)';
                this.statusBadge.style.color = '#a3a3c2';
        }
    }
    
    activatePipelineStep(step) {
        // Deactivate all steps
        this.sttStep.classList.remove('active');
        this.llmStep.classList.remove('active');
        this.ttsStep.classList.remove('active');
        this.memoryStep.classList.remove('active');
        
        // Activate specified step
        switch (step) {
            case 'stt':
                this.sttStep.classList.add('active');
                break;
            case 'llm':
                this.llmStep.classList.add('active');
                break;
            case 'tts':
                this.ttsStep.classList.add('active');
                break;
            case 'memory':
                this.memoryStep.classList.add('active');
                break;
        }
    }
    
    resetPipeline() {
        this.sttStep.classList.remove('active');
        this.llmStep.classList.remove('active');
        this.ttsStep.classList.remove('active');
        this.memoryStep.classList.remove('active');
    }
}

// Initialize assistant when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Miss Riverwood Voice Assistant initialized');
    window.assistant = new VoiceAssistant();
});
