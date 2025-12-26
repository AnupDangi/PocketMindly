import sounddevice as sd
import numpy as np
import queue
import threading
import time
from typing import Callable, Optional

class AudioStream:
    """
    Continuous audio streaming from microphone.
    Feeds audio frames to multiple consumers (VAD, STT) simultaneously.
    """
    
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
        
        self._stream = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        
        # Subscribers (VAD, STT, etc.)
        self._subscribers = []
        
        # Ring buffer for pre-roll (500ms of audio before speech detected)
        self._buffer_size = int(sample_rate * 0.5)  # 500ms
        self._ring_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_pos = 0
        
    def subscribe(self, callback: Callable[[np.ndarray], None]):
        """
        Subscribe to audio frames.
        Callback receives: audio_chunk (float32 numpy array)
        """
        self._subscribers.append(callback)
    
    def start(self):
        """Start streaming audio from microphone"""
        if self._running:
            return
        
        self._running = True
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            
            # Convert to float32 mono
            audio = indata[:, 0].astype(np.float32) / 32768.0
            
            # Update ring buffer
            self._update_ring_buffer(audio)
            
            # Don't send to subscribers if paused
            if self._paused:
                return
            
            # Send to all subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(audio.copy())
                except Exception as e:
                    print(f"Error in subscriber: {e}")
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size,
            callback=audio_callback
        )
        
        self._stream.start()
        print(f"🎤 Audio stream started ({self.sample_rate}Hz, {self.frame_duration_ms}ms frames)")
    
    def stop(self):
        """Stop streaming"""
        if not self._running:
            return
        
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        print("🎤 Audio stream stopped")
    
    def pause(self):
        """Pause sending frames to subscribers (for TTS playback)"""
        with self._lock:
            self._paused = True
            print("⏸️  Audio stream paused")
    
    def resume(self):
        """Resume sending frames to subscribers"""
        with self._lock:
            self._paused = False
            print("▶️  Audio stream resumed")
    
    def get_pre_roll(self) -> np.ndarray:
        """Get buffered audio from before speech was detected"""
        with self._lock:
            # Return last 500ms of audio
            return self._ring_buffer.copy()
    
    def _update_ring_buffer(self, audio: np.ndarray):
        """Update the ring buffer with new audio"""
        with self._lock:
            chunk_len = len(audio)
            
            if chunk_len >= self._buffer_size:
                # Chunk is larger than buffer, just keep the end
                self._ring_buffer = audio[-self._buffer_size:]
                self._buffer_pos = 0
            else:
                # Add to ring buffer
                space_left = self._buffer_size - self._buffer_pos
                
                if chunk_len <= space_left:
                    # Fits in remaining space
                    self._ring_buffer[self._buffer_pos:self._buffer_pos + chunk_len] = audio
                    self._buffer_pos += chunk_len
                else:
                    # Wrap around
                    self._ring_buffer[self._buffer_pos:] = audio[:space_left]
                    remaining = chunk_len - space_left
                    self._ring_buffer[:remaining] = audio[space_left:]
                    self._buffer_pos = remaining
                
                # Reset position if we've filled the buffer
                if self._buffer_pos >= self._buffer_size:
                    self._buffer_pos = 0
