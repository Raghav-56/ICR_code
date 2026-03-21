# Interpreting Credit Risk - ICR Code

Code for the manuscript:
Stability, Scalability, and Agreement: A Systematic Comparison of SHAP and LIME for Loan Default Prediction.

## What this repository enforces

- Single-dataset workflow (Give Me Some Credit).
- Leakage-safe split and preprocessing: stratified 70/15/15, train-only preprocessing stats.
- Multi-model benchmark: logistic regression, XGBoost, CatBoost.
- Imbalance strategy comparison inside training: class weights vs SMOTE-inside-CV.
- Platt calibration and threshold selection on validation split.
- Explanation protocol: TreeSHAP for tree models, KernelSHAP for logistic path, LIME with fixed protocol.
- Agreement and stability outputs with bootstrap confidence intervals.
- Scalability timing with repeated runs and median summary.

## System architecture

### 1) End-to-end pipeline

```mermaid
flowchart LR
  subgraph prep["Data Preparation"]
    A["Raw data<br/>(Give Me Some Credit)"]
    B["Preprocess<br/>(70/15/15 split)"]
  end
  
  subgraph train_eval["Model Training & Evaluation"]
    C["Train/Val/Test<br/>(leakage-safe)"]
    D["Train Models<br/>(LR, XGB, CB)"]
    E["Evaluate<br/>(metrics, calibration)"]
  end
  
  subgraph explain_phase["Explanations & Analysis"]
    F["Explain<br/>(SHAP, LIME)"]
    G["Stability<br/>(perturbation)"]
    H["Scalability<br/>(timing)"]
  end

  subgraph output["Paper Artifacts"]
    I["Reports, Figures<br/>& Metrics"]
    J["📎 Export<br/>(paper assets)"]
  end
  
  A --> B
  B --> C
  C --> D
  D --> E
  D --> F
  D --> G
  D --> H
  E --> I
  F --> I
  G --> I
  H --> I
  I --> J
  
  classDef dataNode fill:#3b82f6,stroke:#1e40af,color:#fff,stroke-width:2px
  classDef processNode fill:#10b981,stroke:#047857,color:#fff,stroke-width:2px
  classDef outputNode fill:#f59e0b,stroke:#b45309,color:#fff,stroke-width:2px
  classDef reportNode fill:#8b5cf6,stroke:#6d28d9,color:#fff,stroke-width:2px
  
  class A,C dataNode
  class B,D,E,F,G,H processNode
  class I outputNode
  class J reportNode
```

### 2) Model training and selection

```mermaid
flowchart TD
  subgraph models["Candidate Models"]
    A["Models<br/>(LR, XGB, CB)"]
  end
  
  subgraph tuning["Hyperparameter Tuning"]
    B["Logistic<br/>Regression"]
    C["XGBoost"]
    D["CatBoost"]
    E["CV Tuning<br/>+ Imbalance Strategies<br/>(Class Weights vs SMOTE)"]
  end
  
  subgraph selection["Model Selection & Calibration"]
    F["Select Best<br/>Configuration"]
    G["Platt Calibration<br/>(validation split)"]
    H["F1 Threshold<br/>Selection"]
  end
  
  subgraph registry["Model Registry"]
    I["model_registry<br/>(CSV / JSON)"]
  end
  
  A --> B
  A --> C
  A --> D
  B --> E
  C --> E
  D --> E
  E --> F
  F --> G
  G --> H
  H --> I
  
  classDef modelNode fill:#3b82f6,stroke:#1e40af,color:#fff,stroke-width:2px
  classDef tuneNode fill:#10b981,stroke:#047857,color:#fff,stroke-width:2px
  classDef selectNode fill:#f59e0b,stroke:#b45309,color:#fff,stroke-width:2px
  classDef outputNode fill:#8b5cf6,stroke:#6d28d9,color:#fff,stroke-width:2px
  
  class A modelNode
  class B,C,D,E tuneNode
  class F,G,H selectNode
  class I outputNode
```

### 3) Explanation, agreement, stability

```mermaid
flowchart TD
  subgraph input["Input"]
    A["Trained Primary<br/>Model"]
  end
  
  subgraph explain["Explanation Methods"]
    B["SHAP<br/>(TreeSHAP/KernelSHAP)"]
    C["LIME<br/>(fixed protocol)"]
  end
  
  subgraph agreement["Feature Importance Agreement"]
    D["Jaccard@k<br/>Top-k feature overlap"]
    E["Spearman<br/>Ranking correlation"]
  end
  
  subgraph stability["Robustness & Reliability"]
    F["Perturbation<br/>Robustness<br/>(σ=0.01)"]
    G["Seed Sensitivity<br/>(50 random seeds)"]
  end
  
  subgraph report["Final Report"]
    H["Bootstrap 95% CI<br/>(1000 resamples)"]
  end
  
  A --> B
  A --> C
  B --> D
  C --> D
  D --> H
  B --> F
  C --> G
  F --> H
  G --> H
  
  classDef modelNode fill:#3b82f6,stroke:#1e40af,color:#fff,stroke-width:2px
  classDef explainNode fill:#10b981,stroke:#047857,color:#fff,stroke-width:2px
  classDef metricNode fill:#f59e0b,stroke:#b45309,color:#fff,stroke-width:2px
  classDef stabilityNode fill:#ec4899,stroke:#be185d,color:#fff,stroke-width:2px
  classDef reportNode fill:#8b5cf6,stroke:#6d28d9,color:#fff,stroke-width:2px
  
  class A modelNode
  class B,C explainNode
  class D,E metricNode
  class F,G stabilityNode
  class H reportNode
```

### 4) Artifact graph

```mermaid
flowchart LR
  subgraph models["Model Artifacts"]
    A["model_registry.csv<br/>(configs & scores)"]
    C["model.joblib<br/>(primary model)"]
  end
  
  subgraph metrics_data["Metrics & Analysis Data"]
    B["test_metrics_all.csv<br/>(all models)"]
    D["agreement.json<br/>(Jaccard, Spearman)"]
    E["stability_summary.json<br/>(means & CI)"]
    F["runtime_median.csv<br/>(timing stats)"]
  end
  
  subgraph visuals["Paper Figures"]
    G["calibration_curve.pdf<br/>(model quality)"]
    H["scalability_runtime.pdf<br/>(timing analysis)"]
  end
  
  subgraph export["Final Export"]
    I["Paper Assets<br/>(figures for PDF)"]
  end
  
  A --> B
  A --> C
  C --> D
  C --> E
  C --> F
  B --> G
  F --> H
  G --> I
  H --> I
  
  classDef modelNode fill:#3b82f6,stroke:#1e40af,color:#fff,stroke-width:2px
  classDef dataNode fill:#10b981,stroke:#047857,color:#fff,stroke-width:2px
  classDef visualNode fill:#f59e0b,stroke:#b45309,color:#fff,stroke-width:2px
  classDef exportNode fill:#8b5cf6,stroke:#6d28d9,color:#fff,stroke-width:2px
  
  class A,C modelNode
  class B,D,E,F dataNode
  class G,H visualNode
  class I exportNode
```

## Commands

```bash
uv sync
python main.py run-paper-protocol --config configs/base.yaml
python main.py export-paper-assets --config configs/base.yaml --paper-figures-dir ../figures
```

Stage-wise execution:

```bash
python main.py prepare-data --config configs/base.yaml
python main.py train --config configs/base.yaml
python main.py evaluate --config configs/base.yaml
python main.py explain --config configs/base.yaml
python main.py explain-sweep --config configs/base.yaml
python main.py stability --config configs/base.yaml
python main.py scalability --config configs/base.yaml
python main.py report --config configs/base.yaml
```

## Paper parity checklist

- Split protocol: `data.test_size=0.15`, `data.val_size=0.15`, fixed seed.
- CV protocol: `models.cv_folds=5`, stratified.
- Imbalance protocol: `models.compare_imbalance_strategies=true`.
- Calibration protocol: Platt sigmoid during training and calibration curve in evaluation.
- Agreement protocol: `explain.top_k=3`, `stability.bootstrap_samples=1000`.
- LIME protocol: `lime_num_samples_sweep=[500,1000,5000]`, `lime_seed_count=50`, `lime_kernel_width=sqrt_features`.
- Perturbation protocol: `stability.perturbation_sigma=0.01`.
- Scalability protocol: `scalability.repeats=5`, includes `22500` point.

## Key outputs

- Models: `artifacts/models/model_registry.csv`, `artifacts/models/model_registry.json`, `artifacts/models/model.joblib`.
- Predictive metrics: `artifacts/metrics/test_metrics_all.csv`, `artifacts/metrics/test_metrics.json`.
- Explanations: `artifacts/explanations/shap_local_abs.csv`, `artifacts/explanations/lime_local_abs.csv`, `artifacts/explanations/agreement.json`.
- Stability: `artifacts/stability/lime_seed_stability.csv`, `artifacts/stability/stability_summary.json`.
- Scalability: `artifacts/scalability/runtime_raw.csv`, `artifacts/scalability/runtime_median.csv`.
- Paper figures: `artifacts/reports/figures/calibration_curve.pdf`, `artifacts/reports/figures/scalability_runtime.pdf`.

## Notes for manuscript sync

- Explanations run on full test set when `explain.use_full_test_for_explanations=true`.
- The primary explanation model is controlled by `models.model_name`.
- `export-paper-assets` copies generated PDFs into the folder consumed by `paper.tex`.
