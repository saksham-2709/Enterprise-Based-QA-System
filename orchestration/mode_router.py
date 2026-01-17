from modes.decision import DecisionMode
from modes.qa import QAMode  
from modes.search import SearchMode

class ModeRouter:
    def __init__(self):
        self.modes = {
            "decision": DecisionMode(),
            "qa": QAMode(),
            "search": SearchMode()
        }

    def get(self, mode: str):
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        return self.modes[mode]
