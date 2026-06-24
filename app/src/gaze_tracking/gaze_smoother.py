from collections import deque


class GazeSmoother:
    def __init__(self, window: int = 8):
        self._history = deque(maxlen=window)

    def update(self, direction: str) -> str:
        self._history.append(direction)
        if not self._history:
            return direction
        return max(set(self._history), key=self._history.count)

    def reset(self):
        self._history.clear()
