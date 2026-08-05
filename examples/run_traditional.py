"""Run the Traditional imperative multi-step processor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python examples/run_traditional.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.shared.llm import configure_logging
from src.shared.samples import SAMPLE_FEEDBACK
from src.traditional import build_traditional_processor

console = Console()


def main(feedback: str = SAMPLE_FEEDBACK) -> None:
    configure_logging()
    console.print(Panel.fit("[bold]Traditional LangChain Multi-Step Processor[/bold]"))

    processor = build_traditional_processor()
    report, steps = processor.process(feedback)

    table = Table(title="Step timings")
    table.add_column("Step")
    table.add_column("ms", justify="right")
    for step in steps:
        table.add_row(step.name, f"{step.elapsed_ms:.1f}")
    console.print(table)

    console.print(
        Panel(
            json.dumps(report.model_dump(), indent=2),
            title="AnalysisReport",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
