from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression

from icr.config import PipelineConfig, load_config
from icr.data.clean import fit_cleaning, transform_cleaning
from icr.data.kaggle_download import ensure_kaggle_competition_dataset
from icr.data.load import load_raw_dataframe
from icr.data.split import stratified_split
from icr.data.validate import validate_dataframe
from icr.evaluation.agreement import (
    bootstrap_ci,
    per_instance_jaccard_at_k,
    per_instance_spearman,
    summarize_agreement_with_ci,
)
from icr.evaluation.metrics import (
    compute_binary_metrics,
    predict_with_threshold,
    select_threshold_by_f1,
)
from icr.evaluation.plots import save_calibration_plot, save_runtime_plot
from icr.explanation.lime_explainer import explain_with_lime, explain_with_lime_local
from icr.explanation.shap_explainer import (
    explain_with_shap_local,
    summarize_local_attributions,
)
from icr.models.train import train_model
from icr.reporting.build_report import build_report
from icr.scalability.benchmark import bench_callable, summarize_bench_medians
from icr.stability.perturbation import perturb_numeric
from icr.utils.io import ensure_dir, load_model, save_dataframe, save_json, save_model
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


def _load_processed(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc = Path(cfg.paths.processed_dir)
    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")
    test = pd.read_csv(proc / "test.csv")
    return train, val, test


def _sample_x_test(x_test: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.explain.use_full_test_for_explanations:
        return x_test.reset_index(drop=True)
    return x_test.sample(
        n=min(cfg.explain.sample_size, len(x_test)),
        random_state=cfg.project.random_seed,
    ).reset_index(drop=True)


def _load_model_registry(cfg: PipelineConfig) -> dict[str, Any]:
    path = _artifacts_root(cfg) / "models" / "model_registry.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_selected_model(cfg: PipelineConfig):
    model_name = cfg.models.model_name.lower()
    candidate_path = _artifacts_root(cfg) / "models" / f"{model_name}.joblib"
    if candidate_path.exists():
        return load_model(candidate_path)
    return load_model(_artifacts_root(cfg) / "models" / "model.joblib")


def _lime_seeds(cfg: PipelineConfig) -> list[int]:
    seeds = cfg.explain.lime_seeds[: cfg.explain.lime_seed_count]
    if len(seeds) >= cfg.explain.lime_seed_count:
        return seeds
    extra = []
    base = 1009
    while len(seeds) + len(extra) < cfg.explain.lime_seed_count:
        extra.append(base)
        base += 13
    return seeds + extra


@app.command("prepare-data")
def prepare_data(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    ensure_kaggle_competition_dataset(cfg)

    raw = load_raw_dataframe(cfg)
    validate_dataframe(raw, cfg)
    splits = stratified_split(raw, cfg)

    cleaning_art = fit_cleaning(splits.train, cfg)
    train_df = transform_cleaning(splits.train, cleaning_art, cfg)
    val_df = transform_cleaning(splits.val, cleaning_art, cfg)
    test_df = transform_cleaning(splits.test, cleaning_art, cfg)

    out_dir = ensure_dir(Path(cfg.paths.processed_dir))
    save_dataframe(out_dir / "train.csv", train_df)
    save_dataframe(out_dir / "val.csv", val_df)
    save_dataframe(out_dir / "test.csv", test_df)

    save_json(
        _artifacts_root(cfg) / "metrics" / "dataset_info.json",
        {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "target": cfg.data.target_col,
            "test_size": cfg.data.test_size,
            "val_size": cfg.data.val_size,
            "seed": cfg.project.random_seed,
        },
    )
    logger.info("Data preparation complete.")


@app.command("download-dataset")
def download_dataset_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
) -> None:
    cfg = _load_cfg(config)
    downloaded = ensure_kaggle_competition_dataset(cfg, required=True)
    logger.info("Dataset download complete: %s", downloaded)


@app.command("train")
def train_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    train_df, val_df, _ = _load_processed(cfg)
    x_val = val_df.drop(columns=[cfg.data.target_col])
    y_val = val_df[cfg.data.target_col]

    model_rows: list[dict[str, Any]] = []
    for model_name in cfg.models.candidate_models:
        artifacts = train_model(train_df, val_df, cfg, model_name=model_name)
        model_path = _artifacts_root(cfg) / "models" / f"{model_name}.joblib"
        save_model(model_path, artifacts.model)

        val_prob = artifacts.model.predict_proba(x_val)[:, 1]
        threshold = select_threshold_by_f1(y_val, val_prob)
        model_rows.append(
            {
                "model_name": model_name,
                "path": str(model_path.as_posix()),
                "selected_strategy": artifacts.selected_strategy,
                "cv_best_score": artifacts.cv_best_score,
                "cv_folds": cfg.models.cv_folds,
                "tuned": cfg.models.tune_hyperparameters,
                "val_threshold": threshold,
                "feature_count": len(artifacts.feature_columns),
            }
        )

    registry_df = pd.DataFrame(model_rows)
    save_dataframe(_artifacts_root(cfg) / "models" / "model_registry.csv", registry_df)
    save_json(
        _artifacts_root(cfg) / "models" / "model_registry.json",
        {"models": model_rows},
    )

    selected_name = cfg.models.model_name.lower()
    selected_row = registry_df.loc[registry_df["model_name"] == selected_name]
    if selected_row.empty:
        raise ValueError(f"Configured primary model '{selected_name}' not found in candidate_models")
    selected_path = Path(selected_row.iloc[0]["path"])
    save_model(_artifacts_root(cfg) / "models" / "model.joblib", load_model(selected_path))

    save_json(
        _artifacts_root(cfg) / "models" / "model_meta.json",
        {
            "model_name": selected_name,
            "candidate_models": cfg.models.candidate_models,
            "features": int(selected_row.iloc[0]["feature_count"]),
            "selected_strategy": str(selected_row.iloc[0]["selected_strategy"]),
            "cv_best_score": (
                None
                if pd.isna(selected_row.iloc[0]["cv_best_score"])
                else float(selected_row.iloc[0]["cv_best_score"])
            ),
            "val_threshold": float(selected_row.iloc[0]["val_threshold"]),
            "cv_folds": cfg.models.cv_folds,
            "tuned": cfg.models.tune_hyperparameters,
        },
    )
    logger.info("Training complete for models: %s", ", ".join(cfg.models.candidate_models))


@app.command("evaluate")
def evaluate_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    _, _, test_df = _load_processed(cfg)

    registry_payload = _load_model_registry(cfg)
    model_entries = registry_payload.get("models", []) if registry_payload else []
    if not model_entries:
        model_entries = [
            {
                "model_name": cfg.models.model_name,
                "path": str((_artifacts_root(cfg) / "models" / "model.joblib").as_posix()),
                "val_threshold": 0.5,
            }
        ]

    x_test = test_df.drop(columns=[cfg.data.target_col])
    y_test = test_df[cfg.data.target_col]

    rows: list[dict[str, float | str]] = []
    calibration_inputs: dict[str, Any] = {}
    for entry in model_entries:
        model_name = str(entry["model_name"])
        model_path = Path(str(entry["path"]))
        threshold = float(entry.get("val_threshold", 0.5))
        model = load_model(model_path)

        y_pred, y_prob = predict_with_threshold(model, x_test, threshold=threshold)
        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        rows.append({"model": model_name, "threshold": threshold, **metrics})
        calibration_inputs[model_name] = y_prob

    metrics_df = pd.DataFrame(rows)
    save_dataframe(_artifacts_root(cfg) / "metrics" / "test_metrics_all.csv", metrics_df)

    selected_name = cfg.models.model_name.lower()
    selected_row = metrics_df.loc[metrics_df["model"] == selected_name]
    if selected_row.empty:
        selected_row = metrics_df.iloc[[0]]
    selected_metrics = selected_row.iloc[0].drop(labels=["model", "threshold"]).to_dict()
    save_json(_artifacts_root(cfg) / "metrics" / "test_metrics.json", selected_metrics)

    save_calibration_plot(
        y_test,
        calibration_inputs,
        _artifacts_root(cfg) / "reports" / "figures" / "calibration_curve.pdf",
    )
    logger.info("Evaluation complete for %d model(s).", len(rows))


@app.command("explain")
def explain_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = _load_selected_model(cfg)
    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])
    sample = _sample_x_test(x_test, cfg)

    shap_local = explain_with_shap_local(
        model,
        x_train,
        sample,
        kernel_background_size=cfg.explain.shap_kernel_background_size,
        kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
    )
    lime_local = explain_with_lime_local(
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

    shap_df = summarize_local_attributions(shap_local, "shap")
    lime_df = summarize_local_attributions(lime_local, "lime")

    save_dataframe(_artifacts_root(cfg) / "explanations" / "shap_importance.csv", shap_df)
    save_dataframe(_artifacts_root(cfg) / "explanations" / "lime_importance.csv", lime_df)
    save_dataframe(_artifacts_root(cfg) / "explanations" / "shap_local_abs.csv", shap_local)
    save_dataframe(_artifacts_root(cfg) / "explanations" / "lime_local_abs.csv", lime_local)

    compare = summarize_agreement_with_ci(
        shap_local,
        lime_local,
        top_k=cfg.explain.top_k,
        n_bootstrap=cfg.stability.bootstrap_samples,
        seed=cfg.project.random_seed,
    )
    save_json(_artifacts_root(cfg) / "explanations" / "agreement.json", compare)
    logger.info("Explanation complete: %s", compare)


@app.command("explain-sweep")
def explain_sweep_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)
    model = _load_selected_model(cfg)

    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])
    sample = _sample_x_test(x_test, cfg)

    shap_local = explain_with_shap_local(
        model,
        x_train,
        sample,
        kernel_background_size=cfg.explain.shap_kernel_background_size,
        kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
    )

    rows = []
    for num_samples in cfg.explain.lime_num_samples_sweep:
        lime_local = explain_with_lime_local(
            model,
            x_train,
            sample,
            num_features=cfg.explain.lime_num_features,
            num_samples=num_samples,
            kernel_width=cfg.explain.lime_kernel_width,
            feature_selection=cfg.explain.lime_feature_selection,
            discretize_continuous=cfg.explain.lime_discretize_continuous,
            random_state=cfg.project.random_seed,
        )
        summary = summarize_agreement_with_ci(
            shap_local,
            lime_local,
            top_k=cfg.explain.top_k,
            n_bootstrap=cfg.stability.bootstrap_samples,
            seed=cfg.project.random_seed,
        )
        rows.append({"lime_num_samples": num_samples, **summary})

    save_dataframe(_artifacts_root(cfg) / "explanations" / "lime_num_samples_sweep.csv", pd.DataFrame(rows))
    logger.info("LIME num_samples sweep complete.")


@app.command("stability")
def stability_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = _load_selected_model(cfg)
    x_train = train_df.drop(columns=[cfg.data.target_col])
    x_test = test_df.drop(columns=[cfg.data.target_col])
    base = _sample_x_test(x_test, cfg)
    train_std = x_train.std(numeric_only=True)

    base_lime = explain_with_lime_local(
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

    shap_base = explain_with_shap_local(
        model,
        x_train,
        base,
        kernel_background_size=cfg.explain.shap_kernel_background_size,
        kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
    )

    shap_repeat_means = []
    lime_repeat_means = []
    for repeat_idx in range(cfg.stability.perturbation_repeats):
        repeat_seed = cfg.project.random_seed + repeat_idx
        perturbed = perturb_numeric(
            base,
            cfg.stability.perturbation_sigma,
            repeat_seed,
            reference_std=train_std,
        )
        shap_perturbed = explain_with_shap_local(
            model,
            x_train,
            perturbed,
            kernel_background_size=cfg.explain.shap_kernel_background_size,
            kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
        )
        lime_perturbed = explain_with_lime_local(
            model,
            x_train,
            perturbed,
            num_features=cfg.explain.lime_num_features,
            num_samples=cfg.explain.lime_num_samples,
            kernel_width=cfg.explain.lime_kernel_width,
            feature_selection=cfg.explain.lime_feature_selection,
            discretize_continuous=cfg.explain.lime_discretize_continuous,
            random_state=repeat_seed,
        )

        shap_perturb_j = per_instance_jaccard_at_k(
            shap_base,
            shap_perturbed,
            cfg.explain.top_k,
        )
        lime_perturb_j = per_instance_jaccard_at_k(
            base_lime,
            lime_perturbed,
            cfg.explain.top_k,
        )
        shap_repeat_means.append(float(pd.Series(shap_perturb_j).mean()))
        lime_repeat_means.append(float(pd.Series(lime_perturb_j).mean()))

    seed_rows = []
    for seed in _lime_seeds(cfg):
        rep = explain_with_lime_local(
            model,
            x_train,
            base,
            num_features=cfg.explain.lime_num_features,
            num_samples=cfg.explain.lime_num_samples,
            kernel_width=cfg.explain.lime_kernel_width,
            feature_selection=cfg.explain.lime_feature_selection,
            discretize_continuous=cfg.explain.lime_discretize_continuous,
            random_state=seed,
        )
        j_vals = per_instance_jaccard_at_k(base_lime, rep, cfg.explain.top_k)
        s_vals = per_instance_spearman(base_lime, rep)
        seed_rows.append(
            {
                "seed": seed,
                "jaccard_at_k": float(pd.Series(j_vals).mean()),
                "spearman": float(pd.Series(s_vals).mean()),
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    save_dataframe(_artifacts_root(cfg) / "stability" / "lime_seed_stability.csv", seed_df)

    stability_summary = {
        "lime_seed_jaccard_mean": float(seed_df["jaccard_at_k"].mean()),
        "lime_seed_jaccard_ci_low": float(
            bootstrap_ci(
                seed_df["jaccard_at_k"].tolist(),
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_low"]
        ),
        "lime_seed_jaccard_ci_high": float(
            bootstrap_ci(
                seed_df["jaccard_at_k"].tolist(),
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_high"]
        ),
        "lime_seed_spearman_mean": float(seed_df["spearman"].mean()),
        "lime_seed_spearman_ci_low": float(
            bootstrap_ci(
                seed_df["spearman"].tolist(),
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_low"]
        ),
        "lime_seed_spearman_ci_high": float(
            bootstrap_ci(
                seed_df["spearman"].tolist(),
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_high"]
        ),
        "perturbation_repeats": cfg.stability.perturbation_repeats,
        "perturbation_shap_jaccard_mean": float(pd.Series(shap_repeat_means).mean()),
        "perturbation_shap_jaccard_ci_low": float(
            bootstrap_ci(
                shap_repeat_means,
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_low"]
        ),
        "perturbation_shap_jaccard_ci_high": float(
            bootstrap_ci(
                shap_repeat_means,
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_high"]
        ),
        "perturbation_lime_jaccard_mean": float(pd.Series(lime_repeat_means).mean()),
        "perturbation_lime_jaccard_ci_low": float(
            bootstrap_ci(
                lime_repeat_means,
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_low"]
        ),
        "perturbation_lime_jaccard_ci_high": float(
            bootstrap_ci(
                lime_repeat_means,
                n_boot=cfg.stability.bootstrap_samples,
                seed=cfg.project.random_seed,
            )["ci_high"]
        ),
    }
    save_json(_artifacts_root(cfg) / "stability" / "stability_summary.json", stability_summary)
    logger.info("Stability analysis complete: %s", stability_summary)


@app.command("scalability")
def scalability_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    train_df, _, test_df = _load_processed(cfg)

    model = _load_selected_model(cfg)
    x_train = train_df.drop(columns=[cfg.data.target_col])
    y_train = train_df[cfg.data.target_col]
    x_test = test_df.drop(columns=[cfg.data.target_col])

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    logistic.fit(x_train, y_train)

    def _tree_or_model_shap(subset: pd.DataFrame):
        explain_with_shap_local(
            model,
            x_train,
            subset,
            kernel_background_size=cfg.explain.shap_kernel_background_size,
            kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
        )

    def _kernelshap_logistic(subset: pd.DataFrame):
        explain_with_shap_local(
            logistic,
            x_train,
            subset,
            kernel_background_size=cfg.explain.shap_kernel_background_size,
            kernel_use_kmeans=cfg.explain.shap_kernel_use_kmeans,
        )

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
    rows.extend(
        bench_callable(
            "treeshap_or_model_shap",
            _tree_or_model_shap,
            x_test,
            cfg.scalability.sizes,
            cfg.scalability.repeats,
        )
    )
    rows.extend(
        bench_callable(
            "kernelshap_logistic",
            _kernelshap_logistic,
            x_test,
            cfg.scalability.sizes,
            cfg.scalability.repeats,
        )
    )
    rows.extend(
        bench_callable(
            "lime",
            _lime,
            x_test,
            cfg.scalability.sizes,
            cfg.scalability.repeats,
        )
    )

    raw_df = pd.DataFrame([r.__dict__ for r in rows])
    med_df = summarize_bench_medians(rows)
    save_dataframe(_artifacts_root(cfg) / "scalability" / "runtime_raw.csv", raw_df)
    save_dataframe(_artifacts_root(cfg) / "scalability" / "runtime_median.csv", med_df)
    save_runtime_plot(med_df, _artifacts_root(cfg) / "reports" / "figures" / "scalability_runtime.pdf")
    logger.info("Scalability benchmark complete.")


@app.command("report")
def report_cmd(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    cfg = _load_cfg(config)
    art = _artifacts_root(cfg)

    metrics_path = art / "metrics" / "test_metrics_all.csv"
    metrics_single_path = art / "metrics" / "test_metrics.json"
    agreement_path = art / "explanations" / "agreement.json"
    runtime_path = art / "scalability" / "runtime_median.csv"

    metrics_df = pd.DataFrame([{}])
    explain_df = pd.DataFrame([{}])
    scal_df = pd.DataFrame([{}])

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    elif metrics_single_path.exists():
        metrics_df = pd.read_json(metrics_single_path, typ="series").to_frame().T
    if agreement_path.exists():
        explain_df = pd.read_json(agreement_path, typ="series").to_frame().T
    if runtime_path.exists():
        scal_df = pd.read_csv(runtime_path)

    build_report(art / "reports" / "summary.md", metrics_df, explain_df, scal_df)
    logger.info("Report generated.")


@app.command("export-paper-assets")
def export_paper_assets_cmd(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
    paper_figures_dir: str = typer.Option("../figures", help="Target figures directory for paper.tex"),
) -> None:
    cfg = _load_cfg(config)
    art = _artifacts_root(cfg)
    target = ensure_dir(Path(paper_figures_dir))

    calibration_src = art / "reports" / "figures" / "calibration_curve.pdf"
    runtime_src = art / "reports" / "figures" / "scalability_runtime.pdf"

    if calibration_src.exists():
        target.joinpath("calibration_curve.pdf").write_bytes(calibration_src.read_bytes())
    if runtime_src.exists():
        target.joinpath("scalability_runtime.pdf").write_bytes(runtime_src.read_bytes())

    logger.info("Paper assets exported to %s", target)


@app.command("run-paper-protocol")
def run_paper_protocol(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    prepare_data(config)
    train_cmd(config)
    evaluate_cmd(config)
    explain_cmd(config)
    explain_sweep_cmd(config)
    stability_cmd(config)
    scalability_cmd(config)
    report_cmd(config)


@app.command("run-all")
def run_all(config: str = typer.Option("configs/base.yaml", help="Path to YAML config")) -> None:
    run_paper_protocol(config)


if __name__ == "__main__":
    app()
