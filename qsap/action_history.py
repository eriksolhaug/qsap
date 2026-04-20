"""
ActionHistory - Undo/Redo system for QSAP
Tracks all major actions and allows navigation through history
"""

from datetime import datetime
from copy import deepcopy


class Action:
    """Represents a single action in the history"""
    
    def __init__(self, action_type, description, state_snapshot, timestamp=None):
        self.action_type = action_type  # 'fit_gaussian', 'fit_voigt', 'fit_continuum', 'clear_fits', etc.
        self.description = description  # Human-readable description
        self.state_snapshot = state_snapshot  # Dictionary containing relevant state at this point
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self):
        return f"Action({self.action_type}: {self.description})"


class ActionHistory:
    """Manages undo/redo history with action tracking"""
    
    def __init__(self, max_history=100):
        self.history = []  # List of Action objects
        self.current_position = -1  # Index in history (-1 = before first action)
        self.max_history = max_history  # Limit history size
    
    def record_action(self, action_type, description, state_snapshot):
        """Record a new action and clear any future history"""
        # Remove any actions after current position (user did something new after undo)
        if self.current_position < len(self.history) - 1:
            self.history = self.history[:self.current_position + 1]
        
        # Create and add new action
        action = Action(action_type, description, deepcopy(state_snapshot))
        self.history.append(action)
        self.current_position += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_position -= 1
    
    def undo(self):
        """Move to previous action in history"""
        if self.current_position > 0:
            self.current_position -= 1
            return self.get_current_state()
        return None
    
    def redo(self):
        """Move to next action in history"""
        if self.current_position < len(self.history) - 1:
            self.current_position += 1
            return self.get_current_state()
        return None
    
    def goto_action(self, index):
        """Jump to a specific action in history"""
        if 0 <= index < len(self.history):
            self.current_position = index
            return self.get_current_state()
        return None
    
    def get_current_state(self):
        """Get the state snapshot of the current position"""
        if 0 <= self.current_position < len(self.history):
            return self.history[self.current_position].state_snapshot
        return None
    
    def get_current_action(self):
        """Get the current action object"""
        if 0 <= self.current_position < len(self.history):
            return self.history[self.current_position]
        return None
    
    def is_at_start(self):
        """Check if at the beginning of history"""
        return self.current_position <= 0
    
    def is_at_end(self):
        """Check if at the end of history"""
        return self.current_position >= len(self.history) - 1
    
    def can_undo(self):
        """Check if undo is possible"""
        return self.current_position > 0
    
    def can_redo(self):
        """Check if redo is possible"""
        return self.current_position < len(self.history) - 1
    
    def get_history_list(self):
        """Get list of all actions for UI display"""
        return self.history
    
    def get_current_position(self):
        """Get current position in history"""
        return self.current_position
    
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self.current_position = -1
