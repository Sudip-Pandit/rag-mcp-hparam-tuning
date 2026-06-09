"""Example MCP tools for the support-ticket agent (Part 5.2).

Each tool declares its risk level. Destructive tools (close/delete) are gated.
These are local stand-ins; in production they'd call your ticketing API.
"""

from typing import Callable, Dict
from .config import ActionRisk

# In-memory ticket store for the demo.
_TICKETS: Dict[str, dict] = {
    "T-100": {"status": "open", "comments": [], "subject": "Login fails"},
}


def get_ticket(ticket_id: str) -> dict:
    return _TICKETS.get(ticket_id, {"error": "not found"})


def add_comment(ticket_id: str, comment: str) -> dict:
    t = _TICKETS.get(ticket_id)
    if not t:
        return {"error": "not found"}
    t["comments"].append(comment)
    return {"ok": True, "comments": len(t["comments"])}


def escalate_ticket(ticket_id: str) -> dict:
    t = _TICKETS.get(ticket_id)
    if not t:
        return {"error": "not found"}
    t["status"] = "escalated"
    return {"ok": True, "status": "escalated"}


def close_ticket(ticket_id: str) -> dict:  # destructive
    t = _TICKETS.get(ticket_id)
    if not t:
        return {"error": "not found"}
    t["status"] = "closed"
    return {"ok": True, "status": "closed"}


def delete_ticket(ticket_id: str) -> dict:  # destructive
    return {"ok": bool(_TICKETS.pop(ticket_id, None))}


# Registry: name -> (callable, risk level)
TOOL_REGISTRY: Dict[str, Dict] = {
    "get_ticket": {"fn": get_ticket, "risk": ActionRisk.READ_ONLY},
    "add_comment": {"fn": add_comment, "risk": ActionRisk.LOW_RISK},
    "escalate_ticket": {"fn": escalate_ticket, "risk": ActionRisk.MEDIUM_RISK},
    "close_ticket": {"fn": close_ticket, "risk": ActionRisk.HIGH_RISK},
    "delete_ticket": {"fn": delete_ticket, "risk": ActionRisk.CRITICAL},
}
