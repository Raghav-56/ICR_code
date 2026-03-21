from __future__ import annotations

import sys
from pathlib import Path

import typer

# Support `python main.py` in src-layout projects without requiring editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icr import cli as icr_cli

app = typer.Typer(
    help=(
        "ICR command center for the SHAP vs LIME credit-risk study.\n\n"
        "Use the grouped commands below, or run the full "
        "paper-aligned pipeline via:\n"
        "  python main.py run-paper-protocol --config configs/base.yaml"
    ),
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(
    icr_cli.app,
    name="pipeline",
    help="All low-level pipeline commands",
)


@app.command("run-paper-protocol")
def run_paper_protocol(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Run the complete manuscript protocol end-to-end."""
    icr_cli.run_paper_protocol(config)


@app.command("prepare-data")
def prepare_data(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Prepare cleaned and split datasets."""
    icr_cli.prepare_data(config)


@app.command("train")
def train(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Train, compare imbalance strategy, and calibrate model."""
    icr_cli.train_cmd(config)


@app.command("evaluate")
def evaluate(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Evaluate predictions and export calibration curve."""
    icr_cli.evaluate_cmd(config)


@app.command("explain")
def explain(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Generate SHAP/LIME explanations and agreement CIs."""
    icr_cli.explain_cmd(config)


@app.command("explain-sweep")
def explain_sweep(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Run LIME num_samples sweep against fixed SHAP baseline."""
    icr_cli.explain_sweep_cmd(config)


@app.command("stability")
def stability(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Run perturbation robustness and LIME seed sensitivity analysis."""
    icr_cli.stability_cmd(config)


@app.command("scalability")
def scalability(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Benchmark runtime scaling and export scalability plot."""
    icr_cli.scalability_cmd(config)


@app.command("report")
def report(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
) -> None:
    """Build markdown summary report from generated artifacts."""
    icr_cli.report_cmd(config)


@app.command("export-paper-assets")
def export_paper_assets(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config."),
    paper_figures_dir: str = typer.Option(
        "../figures",
        help="Destination directory for paper figures.",
    ),
) -> None:
    """Copy generated figures into the folder referenced by paper.tex."""
    icr_cli.export_paper_assets_cmd(config, paper_figures_dir)


if __name__ == "__main__":
    app()
