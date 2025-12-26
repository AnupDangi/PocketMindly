#!/usr/bin/env python3
"""
PocketMindly - FULL Streaming Voice Assistant
Complete streaming pipeline: VAD + STT both run in real-time.
"""

import time
import numpy as np
import scipy.io.wavfile as wav
import asyncio
from core.audio_stream import AudioStream
from core.vad import SileroVAD
from core.state_machine import StateMachine, State
from core.stt import PocketSTT
from core.llm import PocketLLM
from core.audio import speak_text
from tools.web_search import AsyncWebSearchTool

class FullStreamingAssistant:
    def __init__(self):
        print("🚀 Initializing PocketMindly (Full Streaming)...")
        
        # Core components
        self.audio_stream = AudioStream(sample_rate=16000, frame_duration_ms=30)
        self.vad = SileroVAD()
        self.state_machine = StateMachine()
        self.stt = PocketSTT()
        self.llm = PocketLLM()
        self.web_tool = AsyncWebSearchTool()
        
        # State tracking
        self.silence_start = None
        self.silence_threshold = 1.5  # 1.5 seconds (more forgiving)
        self.recording_buffer = []
        self.is_running = True
        
        # Streaming STT state
        self.stt_buffer_size = 1.0  # Process STT every 1 second of audio
        self.stt_buffer = []
        self.last_partial_text = ""
        
        # Subscribe to audio frames
        self.audio_stream.subscribe(self.on_audio_frame)
        
        print("✅ System Ready\n")
    
    def on_audio_frame(self, audio_chunk: np.ndarray):
        """Process each audio frame from the stream"""
        current_state = self.state_machine.state
        
        # Only process when listening or recording
        if current_state not in [State.LISTENING, State.RECORDING]:
            return
        
        # Run VAD
        result = self.vad.process_frame(audio_chunk)
        is_speech = result['is_speech']
        event = result['event']
        
        # Handle speech start
        if event == 'speech_start':
            if current_state == State.LISTENING:
                print("\n🎤 Recording...")
                self.state_machine.transition(State.RECORDING)
                self.recording_buffer = []
                self.stt_buffer = []
                self.last_partial_text = ""
                self.silence_start = None
        
        # Buffer audio if recording
        if current_state == State.RECORDING:
            self.recording_buffer.append(audio_chunk)
            self.stt_buffer.append(audio_chunk)
            
            # Process STT streaming (every 1 second)
            buffer_duration = len(self.stt_buffer) * len(audio_chunk) / 16000
            if buffer_duration >= self.stt_buffer_size:
                self.process_partial_stt()
            
            # Track silence for end detection
            if not is_speech:
                if self.silence_start is None:
                    self.silence_start = time.time()
                else:
                    silence_duration = time.time() - self.silence_start
                    if silence_duration >= self.silence_threshold:
                        self.on_speech_end()
            else:
                self.silence_start = None
    
    def process_partial_stt(self):
        """Process accumulated audio for partial transcript"""
        # DISABLED: Partial transcripts are too noisy and distracting
        # They cause more confusion than help
        # Final transcript on speech end is sufficient
        return
        
        if not self.stt_buffer:
            return
        
        # Concatenate buffer
        audio_data = np.concatenate(self.stt_buffer)
        
        # Get partial transcript
        result = self.stt.transcribe_stream(audio_data, sample_rate=16000)
        partial_text = result['text']
        
        # Only show if different from last
        if partial_text and partial_text != self.last_partial_text:
            print(f"\r💬 {partial_text}...", end="", flush=True)
            self.last_partial_text = partial_text
        
        # Keep only last 0.5s for context
        keep_samples = int(16000 * 0.5)
        if len(audio_data) > keep_samples:
            self.stt_buffer = [audio_data[-keep_samples:]]
        else:
            self.stt_buffer = []
    
    def on_speech_end(self):
        """Handle end of speech - finalize transcript"""
        if self.state_machine.state != State.RECORDING:
            return
        
        self.state_machine.transition(State.PROCESSING)
        
        # Save full audio for final transcription
        audio_data = np.concatenate(self.recording_buffer)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav.write("input.wav", 16000, audio_int16)
        
        # Final transcription
        print("\n💭 Finalizing...", end="", flush=True)
        user_text = self.stt.transcribe("input.wav")
        
        if not user_text:
            print(" [No speech detected]")
            self.reset_to_listening()
            return
        
        print(f"\n👤 You: {user_text}")
        
        # Process with LLM
        self.state_machine.transition(State.THINKING)
        self.process_with_llm(user_text)
    
    async def process_with_llm_async(self, user_text: str):
        """Process user text with LLM and web search if needed (async)"""
        await asyncio.sleep(0.2)
        
        # RULE-BASED SEARCH DETECTION (because Gemma 2B refuses to output SEARCH)
        # Check if question needs web search
        user_lower = user_text.lower()
        
        # Expanded patterns that indicate need for search
        search_keywords = [
            # Question words about people/things
            "who is", "who's", "what is", "what's", "where is", "where's",
            # Time-sensitive
            "latest", "current", "today", "recent", "news",
            # Information requests
            "tell me about", "information about", "find", "search",
            # Specific entities (expand as needed)
            "modi", "musk", "trump", "biden", "china", "russia", "india"
        ]
        
        needs_search = any(keyword in user_lower for keyword in search_keywords)
        
        if needs_search:
            # Extract search query
            search_query = user_text
            for prefix in ["who is ", "who's ", "what is ", "what's ", "tell me about ", "give me "]:
                if prefix in user_lower:
                    search_query = user_text.lower().replace(prefix, "").strip()
                    break
            
            print(f"🔍 Auto-search triggered: {search_query}")
            
            # Run async search
            search_context = await self.web_tool.get_context(search_query)
            
            # DEBUG: Show search context
            print(f"[DEBUG] Search context length: {len(search_context)} chars")
            if len(search_context) > 100:
                print(f"[DEBUG] Search context preview: {search_context[:200]}...")
            
            # Generate answer using search context (in executor)
            print("🤔 Generating answer from search...")
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, 
                self.llm.generate_response_with_search, 
                user_text, 
                search_context
            )
        else:
            # Normal LLM response
            print("🤔 Thinking...")
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(None, self.llm.generate_response, user_text)
            print(f"[DEBUG] LLM returned: '{response_text}'")
        
        print(f"🤖 AI: {response_text}\n")
        
        # Speak response
        self.state_machine.transition(State.SPEAKING)
        self.audio_stream.pause()
        
        # Run TTS in executor (blocking)
        await loop.run_in_executor(None, speak_text, response_text)
        
        self.audio_stream.resume()
        self.reset_to_listening()
    
    def process_with_llm(self, user_text: str):
        """Wrapper to run async processing"""
        asyncio.run(self.process_with_llm_async(user_text))
    
    def reset_to_listening(self):
        """Reset state for next utterance"""
        self.vad.reset_for_new_utterance()
        self.recording_buffer = []
        self.stt_buffer = []
        self.last_partial_text = ""
        self.silence_start = None
        self.state_machine.transition(State.LISTENING)
        print("👂 Listening...")
    
    def run(self):
        """Start the voice assistant"""
        print("=" * 60)
        print("PocketMindly - Full Streaming Mode")
        print("You'll see partial transcripts as you speak!")
        print("Press Ctrl+C to exit")
        print("=" * 60)
        print()
        
        # Start listening
        self.state_machine.transition(State.LISTENING)
        self.audio_stream.start()
        
        print("👂 Listening...")
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            self.audio_stream.stop()

if __name__ == "__main__":
    assistant = FullStreamingAssistant()
    assistant.run()
