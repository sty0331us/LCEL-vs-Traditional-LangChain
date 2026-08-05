"""Side-by-side comparison of Traditional chaining vs Modern LCEL."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.lcel import build_lcel_processor
from src.shared.llm import configure_logging
from src.shared.samples import SAMPLE_FEEDBACK
from src.traditional import build_traditional_processor

app = typer.Typer(
    add_completion=False,
    help="Compare Traditional LangChain chaining vs Modern LCEL.",
)
console = Console()


@app.command()
def compare(
    feedback: Annotated[
        str | None,
        typer.Option("--feedback", "-f", help="Override sample customer feedback"),
    ] = None,
    show_json: Annotated[
        bool,
        typer.Option("--json", help="Print full AnalysisReport JSON for both paths"),
    ] = False,
) -> None:
    """Run both pipelines on the same input and print a comparison table."""
    configure_logging()
    text = feedback or SAMPLE_FEEDBACK

    console.print(
        Panel.fit(
            "[bold]LCEL vs Traditional LangChain[/bold]\n"
            "Same prompts · same schemas · different composition style",
            border_style="magenta",
        )
    )

    # Traditional
    t0 = time.perf_counter()
    traditional = build_traditional_processor()
    trad_report, trad_steps = traditional.process(text)
    trad_total = (time.perf_counter() - t0) * 1000

    # LCEL
    t1 = time.perf_counter()
    lcel = build_lcel_processor()
    lcel_report, lcel_traces = lcel.process(text)
    lcel_total = (time.perf_counter() - t1) * 1000

    summary = Table(title="Comparison summary", show_lines=True)
    summary.add_column("Dimension")
    summary.add_column("Traditional", style="cyan")
    summary.add_column("LCEL", style="green")
    summary.add_row("Composition", "Imperative Python steps", "Declarative `|` graph")
    summary.add_row("Data plumbing", "Manual dict hand-offs", "assign / Passthrough")
    summary.add_row("Async / batch", "Write it yourself", "ainvoke / batch built-in")
    summary.add_row("Streaming", "Per-step wiring", "Native on the Runnable")
    summary.add_row("Total latency (ms)", f"{trad_total:.1f}", f"{lcel_total:.1f}")
    summary.add_row(
        "Priority score",
        str(trad_report.priority_score),
        str(lcel_report.priority_score),
    )
    summary.add_row("Owner team", trad_report.owner_team, lcel_report.owner_team)
    summary.add_row(
        "Sentiment",
        trad_report.sentiment.overall_sentiment,
        lcel_report.sentiment.overall_sentiment,
    )
    console.print(summary)

    steps = Table(title="Traditional step breakdown")
    steps.add_column("Step")
    steps.add_column("ms", justify="right")
    for step in trad_steps:
        steps.add_row(step.name, f"{step.elapsed_ms:.1f}")
    console.print(steps)

    if show_json:
        console.print(
            Panel(
                json.dumps(trad_report.model_dump(), indent=2),
                title="Traditional AnalysisReport",
                border_style="cyan",
            )
        )
        console.print(
            Panel(
                json.dumps(lcel_report.model_dump(), indent=2),
                title="LCEL AnalysisReport",
                border_style="green",
            )
        )
        _ = lcel_traces  # reserved for future per-node tracing demos


if __name__ == "__main__":
    app()
