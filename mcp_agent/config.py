"""MCP agent configuration (Part 2).

Tool-level confidence gating + agent-level planning/memory/budget settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from enum import Enum


class ActionRisk(Enum):
    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium"
    HIGH_RISK = "high"
    CRITICAL = "critical"


_RISK_ORDER = ["read_only", "low_risk", "medium", "high", "critical"]


@dataclass
class MCPToolConfig:
    tool_name: str
    score_threshold: float = 0.65
    requires_confirmation: bool = False
    max_calls_per_turn: int = 3
    timeout_seconds: float = 10.0
    is_destructive: bool = False

    def effective_threshold(self) -> float:
        # Destructive tools demand higher confidence (2.1).
        return 0.85 if self.is_destructive else self.score_threshold


@dataclass
class GuardrailConfig:
    approval_required_above: ActionRisk = ActionRisk.MEDIUM_RISK
    max_api_calls_per_session: int = 100
    max_tokens_per_session: int = 500_000
    max_cost_usd_per_session: float = 2.00
    allowed_tools: Optional[Set[str]] = None
    blocked_tools: Set[str] = field(default_factory=set)
    redact_pii_in_logs: bool = True
    log_all_tool_calls: bool = True

    def is_action_allowed(self, tool_name: str, risk: ActionRisk) -> bool:
        if tool_name in self.blocked_tools:
            return False
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False
        return True

    def needs_approval(self, risk: ActionRisk) -> bool:
        return _RISK_ORDER.index(risk.value) >= _RISK_ORDER.index(
            self.approval_required_above.value
        )


class MCPAgentConfig:
    def __init__(
        self,
        tool_configs: Dict[str, MCPToolConfig],
        global_threshold: float = 0.65,
        routing_confidence: float = 0.80,
        planning_depth: int = 3,
        max_steps: int = 15,
        execution_budget_tokens: int = 50_000,
        memory_window_turns: int = 10,
        guardrails: Optional[GuardrailConfig] = None,
    ):
        self.tool_configs = tool_configs
        self.global_threshold = global_threshold
        self.routing_confidence = routing_confidence
        self.planning_depth = planning_depth
        self.max_steps = max_steps
        self.execution_budget_tokens = execution_budget_tokens
        self.memory_window_turns = memory_window_turns
        self.guardrails = guardrails or GuardrailConfig()
