# context_state.py – new module
from collections import deque
from typing import Deque, Tuple

class ChatState:
    def __init__(self, window_size: int = 8):
        self.window: Deque[Tuple[str, str]] = deque(maxlen=window_size)
        self.summary: str = ""
