import os
from fastmcp import FastMCP
from src.tools.metrics import MetricsEvaluator

mcp = FastMCP("metrics_server", version ="1.0.0")

_evaluator = None


def _get_evaluator() -> MetricsEvaluator:
    """Return the metrics evaluator, initializing it lazily on first use."""
    global _evaluator
    if _evaluator is None:
        _evaluator = MetricsEvaluator()
    return _evaluator


@mcp.tool()
def calculate_metrics(
    complex_text: str,
    current_simplified_text: str,
    reference_text: str,
) -> dict:
    """Return simplification quality and readability metrics for a simplified text."""
    
    return _get_evaluator().calc_simplification_metrics(
        complex_text=complex_text,
        current_simplified_text=current_simplified_text,
        reference_text=reference_text
    )

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("MCP_METRICS_PORT", os.getenv("PORT", "8020")))
    mcp.run(transport="http", host=host, port=port)