import threading
from enum import Enum, auto
from typing import Callable, Optional
import time

class State(Enum):
    """Voice assistant states"""
    IDLE = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()
    THINKING = auto()
    SPEAKING = auto()

class StateMachine:
    """
    Thread-safe state machine for voice assistant.
    Manages state transitions and callbacks.
    """
    
    def __init__(self):
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._callbacks = {}
        
    @property
    def state(self) -> State:
        """Get current state (thread-safe)"""
        with self._lock:
            return self._state
    
    def transition(self, new_state: State) -> bool:
        """
        Transition to a new state.
        Returns True if transition was valid, False otherwise.
        """
        with self._lock:
            old_state = self._state
            
            # Validate transition
            if not self._is_valid_transition(old_state, new_state):
                print(f"❌ Invalid transition: {old_state.name} -> {new_state.name}")
                return False
            
            self._state = new_state
            print(f"🔄 State: {old_state.name} -> {new_state.name}")
            
            # Trigger callbacks
            self._trigger_callbacks(old_state, new_state)
            
            return True
    
    def _is_valid_transition(self, old: State, new: State) -> bool:
        """Check if state transition is valid"""
        valid_transitions = {
            State.IDLE: [State.LISTENING],
            State.LISTENING: [State.RECORDING, State.IDLE],
            State.RECORDING: [State.PROCESSING, State.IDLE],
            State.PROCESSING: [State.THINKING, State.LISTENING, State.IDLE],
            State.THINKING: [State.SPEAKING, State.IDLE],
            State.SPEAKING: [State.IDLE, State.LISTENING],
        }
        
        return new in valid_transitions.get(old, [])
    
    def on_transition(self, from_state: State, to_state: State, callback: Callable):
        """Register a callback for a specific state transition"""
        key = (from_state, to_state)
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    def _trigger_callbacks(self, old_state: State, new_state: State):
        """Trigger callbacks for this transition"""
        key = (old_state, new_state)
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in callback: {e}")
    
    def reset(self):
        """Reset to IDLE state"""
        with self._lock:
            self._state = State.IDLE
            print("🔄 State: RESET -> IDLE")
