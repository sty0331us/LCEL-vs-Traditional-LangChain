"""Run the Modern LCEL multi-step pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.lcel import build_lcel_processor
from src.shared.llm import configure_logging
from src.shared.samples import SAMPLE_FEEDBACK

console = Console()


def main(feedback: str = SAMPLE_FEEDBACK) -> None:
    configure_logging()
    console.print(Panel.fit("[bold]Modern LCEL Multi-Step Pipeline[/bold]"))

    processor = build_lcel_processor()
    report, traces = processor.process(feedback)

    table = Table(title="Pipeline timings")
    table.add_column("Stage")
    table.add_column("ms", justify="right")
    for trace in traces:
        table.add_row(trace.name, f"{trace.elapsed_ms:.1f}")
    console.print(table)

    console.print(
        Panel(
            json.dumps(report.model_dump(), indent=2),
            title="AnalysisReport",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
