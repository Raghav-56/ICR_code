from __future__ import annotations

import logging
from pathlib import Path
from shutil import copy2

from icr.config import PipelineConfig

logger = logging.getLogger(__name__)


def ensure_kaggle_competition_dataset(
    cfg: PipelineConfig,
    required: bool = False,
) -> Path | None:
    """Download and place the configured Kaggle competition file into raw_dir."""
    if not cfg.kagglehub.enabled:
        if required:
            raise ValueError(
                "KaggleHub download is disabled. Set kagglehub.enabled=true in your config."
            )
        return None

    try:
        import kagglehub
    except ImportError as exc:
        raise ModuleNotFoundError(
            "kagglehub is not installed. Run `uv sync` (or install kagglehub) first."
        ) from exc

    raw_dir = Path(cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_file = raw_dir / cfg.data.input_file

    if expected_file.exists() and not cfg.kagglehub.force_download and not cfg.kagglehub.overwrite_existing:
        logger.info("Dataset already present at %s. Skipping KaggleHub download.", expected_file)
        return expected_file

    logger.info(
        "Downloading competition '%s' via kagglehub.competition_download(...)",
        cfg.kagglehub.competition,
    )
    downloaded_path = Path(
        kagglehub.competition_download(
            cfg.kagglehub.competition,
            force_download=cfg.kagglehub.force_download,
        )
    )

    source_file = downloaded_path / cfg.kagglehub.competition_file
    if not source_file.exists():
        raise FileNotFoundError(
            "Downloaded competition folder does not contain the configured file: "
            f"{source_file}. Update kagglehub.competition_file in config."
        )

    if expected_file.exists() and cfg.kagglehub.overwrite_existing:
        expected_file.unlink()

    copy2(source_file, expected_file)
    logger.info("Copied %s to %s", source_file, expected_file)
    return expected_file
