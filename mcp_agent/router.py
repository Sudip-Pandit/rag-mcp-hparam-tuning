"""Tool router with confidence gating (2.1).

Scores each candidate tool against the user message and selects one only if it
clears the tool's effective threshold (higher for destructive tools).
Offline scoring uses keyword affinity; swap in an LLM/classifier in production.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import MCPToolConfig


# Keyword affinities used by the offline scorer.
_TOOL_KEYWORDS = {
    "get_ticket": {"get", "show", "read", "status", "ticket", "look"},
    "add_comment": {"comment", "note", "add", "reply", "update", "say"},
    "escalate_ticket": {"escalate", "urgent", "manager", "priority"},
    "close_ticket": {"close", "resolve", "done", "finished"},
    "delete_ticket": {"delete", "remove", "purge"},
}


@dataclass
class RoutingDecision:
    tool_name: Optional[str]
    confidence: float
    gated: bool  # True if a candidate existed but failed its threshold


def score_tools(message: str) -> Dict[str, float]:
    terms = set(message.lower().split())
    scores = {}
    for tool, kws in _TOOL_KEYWORDS.items():
        hits = len(terms & kws)
        scores[tool] = hits / (len(kws) ** 0.5) if hits else 0.0
    # Normalize to 0..1.
    mx = max(scores.values()) or 1.0
    return {k: round(v / mx, 3) for k, v in scores.items()}


def route(
    message: str,
    tool_configs: Dict[str, MCPToolConfig],
    global_threshold: float = 0.65,
) -> RoutingDecision:
    scores = score_tools(message)
    ranked: List[Tuple[str, float]] = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    if not ranked or ranked[0][1] == 0.0:
        return RoutingDecision(None, 0.0, gated=False)

    name, conf = ranked[0]
    cfg = tool_configs.get(name, MCPToolConfig(name))
    threshold = max(cfg.effective_threshold(), global_threshold) \
        if cfg.is_destructive else cfg.effective_threshold()

    if conf < threshold:
        return RoutingDecision(name, conf, gated=True)
    return RoutingDecision(name, conf, gated=False)


if __name__ == "__main__":
    cfgs = {
        "delete_ticket": MCPToolConfig("delete_ticket", is_destructive=True),
        "add_comment": MCPToolConfig("add_comment"),
    }
    for msg in ["please add a comment to ticket T-100",
                "delete ticket T-100 now"]:
        print(msg, "->", route(msg, cfgs))
