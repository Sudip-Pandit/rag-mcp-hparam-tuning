"""MCP agent loop (Part 2).

Bounded agentic loop with max_steps, token budget, confidence-gated tool
routing, retry, memory window and destructive-action confirmation.

Runs offline with a rule-based planner, or against Anthropic if ANTHROPIC_API_KEY
and the SDK are present (the offline planner keeps the demo deterministic).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import json

from .config import MCPAgentConfig, MCPToolConfig, GuardrailConfig
from .memory import ConversationMemory
from .router import route
from .guardrails import RetryConfig, execute_with_retry, confirm_destructive
from .tools import TOOL_REGISTRY


@dataclass
class AgentResult:
    answer: str
    steps: int = 0
    tokens_used: int = 0
    tool_calls: List[Dict] = field(default_factory=list)
    stopped_reason: str = "end_turn"


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MCPAgent:
    def __init__(
        self,
        config: MCPAgentConfig,
        registry: Optional[Dict] = None,
        retry_cfg: Optional[RetryConfig] = None,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.config = config
        self.registry = registry or TOOL_REGISTRY
        self.retry_cfg = retry_cfg or RetryConfig()
        self.confirm_fn = confirm_fn
        self.memory = ConversationMemory(config.memory_window_turns)

    def run(self, user_message: str, ticket_id: str = "T-100") -> AgentResult:
        self.memory.add("user", user_message)
        result = AgentResult(answer="")
        tokens = 0
        gr: GuardrailConfig = self.config.guardrails

        for step in range(self.config.max_steps):
            result.steps = step + 1
            tokens += _est_tokens(user_message)
            if tokens >= self.config.execution_budget_tokens:
                result.stopped_reason = "budget_exhausted"
                result.answer = "[Budget exhausted before completing the task.]"
                break

            decision = route(
                user_message, self.config.tool_configs, self.config.global_threshold
            )

            if decision.tool_name is None:
                result.answer = "No tool matched; responding directly."
                result.stopped_reason = "no_tool"
                break

            if decision.gated:
                result.answer = (
                    f"Tool '{decision.tool_name}' confidence {decision.confidence} "
                    f"below threshold; not invoked."
                )
                result.stopped_reason = "gated"
                break

            entry = self.registry.get(decision.tool_name)
            if not entry:
                result.answer = f"Unknown tool '{decision.tool_name}'."
                result.stopped_reason = "unknown_tool"
                break

            risk = entry["risk"]
            if not confirm_destructive(
                decision.tool_name, risk, gr, confirm_fn=self.confirm_fn
            ):
                result.answer = (
                    f"Action '{decision.tool_name}' was not approved; nothing changed."
                )
                result.stopped_reason = "denied"
                break

            args = self._build_args(decision.tool_name, ticket_id, user_message)
            tool_out = execute_with_retry(entry["fn"], args, self.retry_cfg)
            result.tool_calls.append(
                {"tool": decision.tool_name, "args": args,
                 "result": tool_out, "confidence": decision.confidence}
            )
            if gr.log_all_tool_calls:
                print(f"[agent] called {decision.tool_name} -> {json.dumps(tool_out)}")

            result.answer = f"Executed {decision.tool_name}: {json.dumps(tool_out)}"
            result.stopped_reason = "end_turn"
            break  # single-tool tasks complete in one step in this demo

        result.tokens_used = tokens
        self.memory.add("assistant", result.answer)
        return result

    @staticmethod
    def _build_args(tool_name: str, ticket_id: str, message: str) -> Dict:
        """Map a routed tool to its required arguments."""
        if tool_name == "add_comment":
            return {"ticket_id": ticket_id, "comment": message}
        return {"ticket_id": ticket_id}


def default_support_agent(confirm_fn: Optional[Callable[[str], bool]] = None) -> MCPAgent:
    """Pre-tuned config matching the article's Support Ticket Agent (5.2)."""
    tool_configs = {
        "get_ticket": MCPToolConfig("get_ticket"),
        "add_comment": MCPToolConfig("add_comment"),
        "escalate_ticket": MCPToolConfig("escalate_ticket"),
        "close_ticket": MCPToolConfig("close_ticket", is_destructive=True,
                                      requires_confirmation=True),
        "delete_ticket": MCPToolConfig("delete_ticket", is_destructive=True,
                                       requires_confirmation=True),
    }
    cfg = MCPAgentConfig(
        tool_configs=tool_configs,
        global_threshold=0.65,
        max_steps=15,
        execution_budget_tokens=50_000,
        memory_window_turns=10,
    )
    return MCPAgent(cfg, confirm_fn=confirm_fn)


if __name__ == "__main__":
    agent = default_support_agent()
    for msg in [
        "please add a comment that we are investigating",
        "delete ticket T-100",                       # gated/denied without approval
    ]:
        print("\nUSER:", msg)
        r = agent.run(msg)
        print("AGENT:", r.answer, "| reason:", r.stopped_reason)
