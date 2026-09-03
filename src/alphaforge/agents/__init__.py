"""Research copilot: a deterministic analyst over the real pipeline outputs."""

from alphaforge.agents.copilot import Briefing, CopilotConfig, ResearchCopilot
from alphaforge.agents.tools import ToolResult, run_tools

__all__ = [
    "ResearchCopilot",
    "CopilotConfig",
    "Briefing",
    "ToolResult",
    "run_tools",
]
