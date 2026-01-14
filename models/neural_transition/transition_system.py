"""
Arc-Standard Transition System for Dependency Parsing

Transitions:
- SHIFT: Move word from buffer to stack
- LEFT-ARC(label): Create arc from stack top to second, pop second
- RIGHT-ARC(label): Create arc from second to stack top, pop stack top

Based on: Chen & Manning (2014) "A Fast and Accurate Dependency Parser using Neural Networks"
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import copy


class ActionType(Enum):
    SHIFT = 0
    LEFT_ARC = 1
    RIGHT_ARC = 2


@dataclass
class Action:
    """A parser action"""
    type: ActionType
    label: Optional[int] = None  # Relation label for arc actions
    
    def __repr__(self):
        if self.type == ActionType.SHIFT:
            return "SHIFT"
        elif self.type == ActionType.LEFT_ARC:
            return f"LEFT-ARC({self.label})"
        else:
            return f"RIGHT-ARC({self.label})"


@dataclass
class ParserState:
    """
    Parser configuration/state.
    
    Attributes:
        stack: List of token indices on the stack (0 = ROOT)
        buffer: List of token indices in the buffer
        heads: Predicted head for each token (-1 = not assigned)
        labels: Predicted label for each token (-1 = not assigned)
        sentence_length: Total number of tokens (including ROOT)
    """
    stack: List[int] = field(default_factory=lambda: [0])  # Start with ROOT
    buffer: List[int] = field(default_factory=list)
    heads: List[int] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    sentence_length: int = 0
    
    @classmethod
    def initial(cls, sentence_length: int) -> 'ParserState':
        """Create initial parser state for a sentence"""
        return cls(
            stack=[0],  # ROOT on stack
            buffer=list(range(1, sentence_length)),  # Words 1..n in buffer
            heads=[-1] * sentence_length,  # -1 = not assigned
            labels=[-1] * sentence_length,
            sentence_length=sentence_length,
        )
    
    def is_terminal(self) -> bool:
        """Check if parsing is complete"""
        return len(self.buffer) == 0 and len(self.stack) == 1
    
    def can_shift(self) -> bool:
        """Check if SHIFT is valid"""
        return len(self.buffer) > 0
    
    def can_left_arc(self) -> bool:
        """Check if LEFT-ARC is valid (can't arc to ROOT)"""
        return len(self.stack) >= 2 and self.stack[-2] != 0
    
    def can_right_arc(self) -> bool:
        """Check if RIGHT-ARC is valid"""
        return len(self.stack) >= 2
    
    def get_stack_top(self, n: int = 0) -> Optional[int]:
        """Get nth element from stack top (0 = top)"""
        if len(self.stack) > n:
            return self.stack[-(n + 1)]
        return None
    
    def get_buffer_front(self, n: int = 0) -> Optional[int]:
        """Get nth element from buffer front"""
        if len(self.buffer) > n:
            return self.buffer[n]
        return None


class ArcStandardTransitionSystem:
    """
    Arc-Standard Transition System.
    
    Actions:
    - SHIFT: buffer[0] -> stack
    - LEFT-ARC(l): stack[-1] -> stack[-2], pop stack[-2]
    - RIGHT-ARC(l): stack[-2] -> stack[-1], pop stack[-1]
    """
    
    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        # Total actions: 1 SHIFT + num_labels LEFT-ARC + num_labels RIGHT-ARC
        self.num_actions = 1 + 2 * num_labels
    
    def get_action_id(self, action: Action) -> int:
        """Convert action to unique ID"""
        if action.type == ActionType.SHIFT:
            return 0
        elif action.type == ActionType.LEFT_ARC:
            return 1 + (action.label or 0)
        else:  # RIGHT_ARC
            return 1 + self.num_labels + (action.label or 0)
    
    def get_action_from_id(self, action_id: int) -> Action:
        """Convert action ID to Action"""
        if action_id == 0:
            return Action(ActionType.SHIFT)
        elif action_id <= self.num_labels:
            return Action(ActionType.LEFT_ARC, action_id - 1)
        else:
            return Action(ActionType.RIGHT_ARC, action_id - 1 - self.num_labels)
    
    def apply(self, state: ParserState, action: Action) -> ParserState:
        """Apply action to state, return new state"""
        new_state = copy.deepcopy(state)
        
        if action.type == ActionType.SHIFT:
            # Move from buffer to stack
            if new_state.can_shift():
                token = new_state.buffer.pop(0)
                new_state.stack.append(token)
        
        elif action.type == ActionType.LEFT_ARC:
            # Arc: stack[-1] -> stack[-2]
            if new_state.can_left_arc():
                head = new_state.stack[-1]
                dep = new_state.stack[-2]
                new_state.heads[dep] = head
                new_state.labels[dep] = action.label or 0
                new_state.stack.pop(-2)
        
        elif action.type == ActionType.RIGHT_ARC:
            # Arc: stack[-2] -> stack[-1]
            if new_state.can_right_arc():
                head = new_state.stack[-2]
                dep = new_state.stack[-1]
                new_state.heads[dep] = head
                new_state.labels[dep] = action.label or 0
                new_state.stack.pop(-1)
        
        return new_state
    
    def get_valid_actions(self, state: ParserState) -> List[Action]:
        """Get list of valid actions for current state"""
        actions = []
        
        if state.can_shift():
            actions.append(Action(ActionType.SHIFT))
        
        if state.can_left_arc():
            for label in range(self.num_labels):
                actions.append(Action(ActionType.LEFT_ARC, label))
        
        if state.can_right_arc():
            for label in range(self.num_labels):
                actions.append(Action(ActionType.RIGHT_ARC, label))
        
        return actions
    
    def get_valid_action_mask(self, state: ParserState) -> List[bool]:
        """Get mask of valid actions (for neural network)"""
        mask = [False] * self.num_actions
        
        if state.can_shift():
            mask[0] = True
        
        if state.can_left_arc():
            for label in range(self.num_labels):
                mask[1 + label] = True
        
        if state.can_right_arc():
            for label in range(self.num_labels):
                mask[1 + self.num_labels + label] = True
        
        return mask


class Oracle:
    """
    Static Oracle for Arc-Standard system.
    
    Given gold parse, generates the correct sequence of actions.
    """
    
    def __init__(self, transition_system: ArcStandardTransitionSystem):
        self.transition_system = transition_system
    
    def get_oracle_action(
        self,
        state: ParserState,
        gold_heads: List[int],
        gold_labels: List[int],
    ) -> Optional[Action]:
        """
        Get the correct action for current state given gold parse.
        
        Returns None if no valid action (shouldn't happen with valid gold parse).
        """
        stack = state.stack
        buffer = state.buffer
        
        if len(stack) < 2:
            # Can only SHIFT
            if state.can_shift():
                return Action(ActionType.SHIFT)
            return None
        
        s0 = stack[-1]  # Stack top
        s1 = stack[-2]  # Second on stack
        
        # Check LEFT-ARC: s0 -> s1 (s0 is head of s1)
        if state.can_left_arc():
            if gold_heads[s1] == s0:
                # Check if s1 has all its dependents
                if self._has_all_dependents(s1, state, gold_heads):
                    return Action(ActionType.LEFT_ARC, gold_labels[s1])
        
        # Check RIGHT-ARC: s1 -> s0 (s1 is head of s0)
        if state.can_right_arc():
            if gold_heads[s0] == s1:
                # Check if s0 has all its dependents
                if self._has_all_dependents(s0, state, gold_heads):
                    return Action(ActionType.RIGHT_ARC, gold_labels[s0])
        
        # Otherwise SHIFT
        if state.can_shift():
            return Action(ActionType.SHIFT)
        
        # Fallback: try RIGHT-ARC if nothing else works
        if state.can_right_arc():
            return Action(ActionType.RIGHT_ARC, gold_labels[s0] if gold_labels[s0] >= 0 else 0)
        
        return None
    
    def _has_all_dependents(
        self,
        token: int,
        state: ParserState,
        gold_heads: List[int],
    ) -> bool:
        """Check if token has collected all its dependents"""
        for i in range(1, state.sentence_length):
            if gold_heads[i] == token:
                # i is a dependent of token
                if state.heads[i] < 0:
                    # i hasn't been assigned a head yet
                    # Check if i is still in buffer or stack
                    if i in state.buffer or i in state.stack:
                        return False
        return True
    
    def get_oracle_sequence(
        self,
        sentence_length: int,
        gold_heads: List[int],
        gold_labels: List[int],
    ) -> Tuple[List[ParserState], List[Action]]:
        """
        Generate complete oracle sequence for a sentence.
        
        Returns:
            states: List of parser states
            actions: List of actions taken
        """
        state = ParserState.initial(sentence_length)
        states = [state]
        actions = []
        
        while not state.is_terminal():
            action = self.get_oracle_action(state, gold_heads, gold_labels)
            if action is None:
                break
            
            actions.append(action)
            state = self.transition_system.apply(state, action)
            states.append(state)
        
        return states[:-1], actions  # Exclude terminal state

