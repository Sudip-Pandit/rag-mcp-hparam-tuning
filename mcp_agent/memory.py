"""2.3 Memory Window & Context Management.

Sliding conversation window with optional summarization of evicted turns.
Treats memory_window_turns as a first-class cost variable.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConversationMemory:
    memory_window_turns: int = 10
    turns: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        # Each "turn" = a user+assistant pair; keep window*2 messages.
        max_msgs = self.memory_window_turns * 2
        if len(self.turns) > max_msgs:
            evicted = self.turns[: len(self.turns) - max_msgs]
            self.turns = self.turns[len(self.turns) - max_msgs :]
            # Cheap extractive summary of evicted content.
            joined = " ".join(t["content"] for t in evicted)
            self.summary = (self.summary + " " + joined).strip()[:500]

    def context(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.summary:
            msgs.append({"role": "system",
                         "content": f"Earlier conversation summary: {self.summary}"})
        msgs.extend(self.turns)
        return msgs


if __name__ == "__main__":
    m = ConversationMemory(memory_window_turns=2)
    for i in range(5):
        m.add("user", f"question {i}")
        m.add("assistant", f"answer {i}")
    print("kept messages:", len(m.turns))
    print("summary:", m.summary[:80])
