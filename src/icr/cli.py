from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from icr.config import PipelineConfig, load_config
from icr.data.clean import fit_cleaning, transform_cleaning
from icr.data.load import load_raw_dataframe
from icr.data.split import stratified_split
from icr.data.validate import validate_dataframe
from icr.evaluation.metrics import (
    compute_binary_metrics,
    predict_with_threshold,
)
from icr.explanation.compare_explanations import rank_correlation, topk_jaccard
from icr.explanation.lime_explainer import explain_with_lime
from icr.explanation.shap_explainer import explain_with_shap
from icr.models.train import train_model
from icr.reporting.build_report import build_report
from icr.scalability.benchmark import bench_callable
from icr.stability.perturbation import perturb_numeric
from icr.utils.io import (
    ensure_dir,
    load_model,
    save_dataframe,
    save_json,
    save_model,
)
from icr.utils.logging import configure_logging
from icr.utils.seed import set_global_seed

app = typer.Typer(help="Interpreting Credit Risk pipeline CLI")
logger = logging.getLogger(__name__)

def _load_cfg(config: str) -> PipelineConfig:
    cfg = load_config(config)
    set_global_seed(cfg.project.random_seed)
    configure_logging()
    return cfg

def _artifacts_root(cfg: PipelineConfig) -> Path:
    return ensure_dir(Path(cfg.paths.artifacts_dir))

def _load_processed(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc = Path(cfg.paths.processed_dir)
    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")
    test = pd.read_csv(proc / "test.csv")
    return train, val, test


@app.command("prepare-data")
def prepare_data(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)

    raw = load_raw_dataframe(cfg)
    validate_dataframe(raw, cfg)
    splits = stratified_split(raw, cfg)

    cleaning_art = fit_cleaning(splits.train, cfg)
    train_df = transform_cleaning(splits.train, cleaning_art, cfg)
    val = transform_cleaning(splits.val, cleaning_art, cfg)
    test = transform_cleaning(splits.test, cleaning_art, cfg)

    out_dir = ensure_dir(Path(cfg.paths.processed_dir))
    save_dataframe(out_dir / "train.csv", train_df)
    save_dataframe(out_dir / "val.csv", val)
    save_dataframe(out_dir / "test.csv", test)

    save_json(_artifacts_root(cfg) / "metrics" / "dataset_info.json", {
        "train_rows": len(train_df),
        "val_rows": len(val),
        "test_rows": len(test),
        "target": cfg.data.target_col,
    })
    logger.info("Data preparation complete.")


@app.command("train")
def train_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    train_df, val_df, _ = _load_processed(cfg)

    artifacts = train_model(train_df, val_df, cfg)
    save_model(_artifacts_root(cfg) / "models" / "model.joblib", artifacts.model)
    save_json(
        _artifacts_root(cfg) / "models" / "model_meta.json",
        {"model_name": cfg.models.model_name, "features": artifacts.feature_columns},
    )
    logger.info("Training complete.")


@app.command("evaluate")
def evaluate_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    _, _, test_df = _load_processed(cfg)

    model = load_model(_artifacts_root(cfg) / "models" / "model.joblib")
    x_test = test_df.drop(columns=[cfg.data.target_col])
    y_test = test_df[cfg.data.target_col]

    y_pred, y_prob = predict_with_threshold(model, x_test)
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    save_json(_artifacts_root(cfg) / "metrics" / "test_metrics.json", metrics)
    logger.info("Evaluation complete: %s", metrics)


@app.command("explain")
def explain_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = load_model(_artifacts_root(cfg) / "models" / "model.joblib")
    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])

    sample = x_test.sample(
        n=min(cfg.explain.sample_size, len(x_test)),
        random_state=cfg.project.random_seed,
    ).reset_index(drop=True)

    shap_df = explain_with_shap(model, sample)
    lime_df = explain_with_lime(
        model,
        x_train,
        sample,
        num_features=cfg.explain.lime_num_features,
        num_samples=cfg.explain.lime_num_samples,
        kernel_width=cfg.explain.lime_kernel_width,
        feature_selection=cfg.explain.lime_feature_selection,
        discretize_continuous=cfg.explain.lime_discretize_continuous,
        random_state=cfg.project.random_seed,
    )

    save_dataframe(_artifacts_root(cfg) / "explanations" / "shap_importance.csv", shap_df)
    save_dataframe(_artifacts_root(cfg) / "explanations" / "lime_importance.csv", lime_df)

    compare = {
        "topk_jaccard": topk_jaccard(shap_df, lime_df, cfg.explain.top_k),
        **rank_correlation(shap_df, lime_df),
    }
    save_json(_artifacts_root(cfg) / "explanations" / "agreement.json", compare)
    logger.info("Explanation complete: %s", compare)


@app.command("stability")
def stability_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = load_model(_artifacts_root(cfg) / "models" / "model.joblib")
    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])
    base = x_test.head(min(cfg.explain.sample_size, len(x_test))).reset_index(drop=True)

    base_lime = explain_with_lime(
        model,
        x_train,
        base,
        num_features=cfg.explain.lime_num_features,
        num_samples=cfg.explain.lime_num_samples,
        kernel_width=cfg.explain.lime_kernel_width,
        feature_selection=cfg.explain.lime_feature_selection,
        discretize_continuous=cfg.explain.lime_discretize_continuous,
        random_state=cfg.project.random_seed,
    )

    rows = []
    for seed in cfg.explain.lime_seeds:
        perturbed = perturb_numeric(base, cfg.stability.perturbation_sigma, seed)
        rep = explain_with_lime(
            model,
            x_train,
            perturbed,
            num_features=cfg.explain.lime_num_features,
            num_samples=cfg.explain.lime_num_samples,
            kernel_width=cfg.explain.lime_kernel_width,
            feature_selection=cfg.explain.lime_feature_selection,
            discretize_continuous=cfg.explain.lime_discretize_continuous,
            random_state=seed,
        )
        corr = rank_correlation(base_lime, rep)
        jac = topk_jaccard(base_lime, rep, cfg.explain.top_k)
        rows.append({"seed": seed, "jaccard": jac, **corr})

    out = pd.DataFrame(rows)
    save_dataframe(_artifacts_root(cfg) / "stability" / "lime_seed_stability.csv", out)
    logger.info("Stability analysis complete.")


@app.command("scalability")
def scalability_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = load_model(_artifacts_root(cfg) / "models" / "model.joblib")
    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])

    def _shap(subset: pd.DataFrame):
        explain_with_shap(model, subset)

    def _lime(subset: pd.DataFrame):
        explain_with_lime(
            model,
            x_train,
            subset,
            num_features=cfg.explain.lime_num_features,
            num_samples=cfg.explain.lime_num_samples,
            kernel_width=cfg.explain.lime_kernel_width,
            feature_selection=cfg.explain.lime_feature_selection,
            discretize_continuous=cfg.explain.lime_discretize_continuous,
            random_state=cfg.project.random_seed,
        )

    rows = []
    rows.extend(bench_callable("shap", _shap, x_test, cfg.scalability.sizes))
    rows.extend(bench_callable("lime", _lime, x_test, cfg.scalability.sizes))

    out = pd.DataFrame([r.__dict__ for r in rows])
    save_dataframe(_artifacts_root(cfg) / "scalability" / "runtime.csv", out)
    logger.info("Scalability benchmark complete.")


@app.command("report")
def report_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    art = _artifacts_root(cfg)

    metrics_path = art / "metrics" / "test_metrics.json"
    agreement_path = art / "explanations" / "agreement.json"
    runtime_path = art / "scalability" / "runtime.csv"

    metrics_df = pd.DataFrame([{}])
    explain_df = pd.DataFrame([{}])
    scal_df = pd.DataFrame([{}])

    if metrics_path.exists():
        metrics_df = pd.read_json(metrics_path, typ="series").to_frame().T
    if agreement_path.exists():
        explain_df = pd.read_json(agreement_path, typ="series").to_frame().T
    if runtime_path.exists():
        scal_df = pd.read_csv(runtime_path)

    build_report(art / "reports" / "summary.md", metrics_df, explain_df, scal_df)
    logger.info("Report generated.")


@app.command("run-all")
def run_all(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    prepare_data(config)
    train_cmd(config)
    evaluate_cmd(config)
    explain_cmd(config)
    stability_cmd(config)
    scalability_cmd(config)
    report_cmd(config)


if __name__ == "__main__":
    app()
