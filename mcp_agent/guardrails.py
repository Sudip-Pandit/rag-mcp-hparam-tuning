"""2.2 / 2.5 Retry strategy + safety guardrails.

Retry with fixed/exponential/jitter backoff, and a confirmation gate for
destructive tools.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import random
import time

from .config import ActionRisk, GuardrailConfig


class RetryStrategy(Enum):
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    JITTER = "jitter"


@dataclass
class RetryConfig:
    strategy: RetryStrategy = RetryStrategy.JITTER
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        if self.strategy == RetryStrategy.FIXED:
            return self.base_delay_s
        exp = self.base_delay_s * (2 ** attempt)
        if self.strategy == RetryStrategy.JITTER:
            exp = exp * (0.5 + random.random() * 0.5)
        return min(exp, self.max_delay_s)


def execute_with_retry(
    tool_fn: Callable, args: dict, retry_cfg: RetryConfig, sleep=time.sleep
):
    last_exc = None
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            return tool_fn(**args)
        except Exception as e:  # narrow to RateLimit/Timeout in production
            last_exc = e
            if attempt == retry_cfg.max_retries:
                raise
            d = retry_cfg.delay(attempt)
            print(f"[retry] {attempt+1}/{retry_cfg.max_retries} after {d:.1f}s: {e}")
            sleep(d)
    raise last_exc  # pragma: no cover


def confirm_destructive(
    tool_name: str,
    risk: ActionRisk,
    guardrails: GuardrailConfig,
    confirm_fn: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Return True if execution may proceed.

    If the action needs approval and no confirm_fn is supplied, deny by default
    (fail-safe). confirm_fn lets you wire in a human-in-the-loop prompt.
    """
    if not guardrails.is_action_allowed(tool_name, risk):
        print(f"[guardrail] '{tool_name}' blocked by policy")
        return False
    if guardrails.needs_approval(risk):
        if confirm_fn is None:
            print(f"[guardrail] '{tool_name}' needs human approval - denied (no handler)")
            return False
        approved = confirm_fn(tool_name)
        print(f"[guardrail] '{tool_name}' approval -> {approved}")
        return approved
    return True


if __name__ == "__main__":
    rc = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, max_retries=3)
    print("delays:", [round(rc.delay(i), 2) for i in range(4)])
    g = GuardrailConfig()
    print("read allowed:", confirm_destructive("get_ticket", ActionRisk.READ_ONLY, g))
    print("delete denied:", confirm_destructive("delete_ticket", ActionRisk.CRITICAL, g))
    print("delete approved:", confirm_destructive(
        "delete_ticket", ActionRisk.CRITICAL, g, confirm_fn=lambda n: True))
